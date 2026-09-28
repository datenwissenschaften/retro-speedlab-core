import argparse
import importlib
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stable_retro
from loguru import logger

STEPS_PER_ROLLOUT = 60
ACTION_CHANGE_PROBABILITY = 0.25
TOP_DEATHS = 15


@dataclass(slots=True)
class Cell:
    emulator_state: bytes
    path: list[int]
    visits: int


@dataclass(frozen=True, slots=True)
class Address:
    location: int
    bucket: int

    @classmethod
    def parse(cls, text: str) -> "Address":
        location, _, bucket = text.partition("/")
        return cls(int(location, 16), int(bucket) if bucket else 1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go-Explore a stable-retro level to map its milestones and pitfalls.")
    parser.add_argument("--game", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--start", type=Path, required=True, help="savestate file, or the literal 'none'")
    parser.add_argument("--actions", required=True, help="module:ATTRIBUTE of the (actions, frames, buttons) table")
    parser.add_argument("--cell", required=True, help="RAM addresses with optional /bucket: 0x4C3,0x4D7/24")
    parser.add_argument("--priority", required=True, help="comma separated weights per cell address")
    parser.add_argument("--lives", required=True, help="RAM address of the lives counter")
    parser.add_argument("--won", required=True, help="RAM address that becomes non-zero when the level is won")
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_action_table(reference: str) -> np.ndarray:
    module_name, attribute = reference.split(":")
    return getattr(importlib.import_module(module_name), attribute)


class Explorer:
    def __init__(self, arguments: argparse.Namespace) -> None:
        self.env = stable_retro.make(arguments.game, state=arguments.state, render_mode=None)
        self.env.reset(seed=arguments.seed)
        if str(arguments.start) != "none":
            self._restore(arguments.start.read_bytes())
        self.actions = load_action_table(arguments.actions)
        self.addresses = tuple(Address.parse(text) for text in arguments.cell.split(","))
        self.priority = tuple(float(weight) for weight in arguments.priority.split(","))
        self.lives, self.won = int(arguments.lives, 16), int(arguments.won, 16)
        self.random = random.Random(arguments.seed)
        self.out = arguments.out
        self.archive = {self._cell(): Cell(self.env.em.get_state(), [], 0)}
        self.milestones: dict[str, dict[str, object]] = {}
        self.deaths: Counter[tuple[int, ...]] = Counter()

    def run(self, seconds: float) -> dict[str, object]:
        started = time.monotonic()
        rollouts = 0
        while time.monotonic() - started < seconds:
            self._rollout(self._choose())
            rollouts += 1
        return {
            "rollouts": rollouts,
            "cells": len(self.archive),
            "maxima": {hex(a.location): max(key[i] for key in self.archive) for i, a in enumerate(self.addresses)},
            "milestones": self.milestones,
            "deaths": [[list(cell), count] for cell, count in self.deaths.most_common(TOP_DEATHS)],
        }

    def _choose(self) -> tuple[int, ...]:
        keys = list(self.archive)
        weights = [self._interest(key) / np.sqrt(self.archive[key].visits + 1) for key in keys]
        return self.random.choices(keys, weights)[0]

    def _interest(self, key: tuple[int, ...]) -> float:
        return 1.0 + sum(weight * value for weight, value in zip(self.priority, key, strict=True))

    def _rollout(self, key: tuple[int, ...]) -> None:
        cell = self.archive[key]
        cell.visits += 1
        self._restore(cell.emulator_state)
        path, lives = list(cell.path), self._ram()[self.lives]
        action = self.random.randrange(len(self.actions))
        for _ in range(STEPS_PER_ROLLOUT):
            if self.random.random() < ACTION_CHANGE_PROBABILITY:
                action = self.random.randrange(len(self.actions))
            terminated = self._play(action)
            path.append(action)
            ram = self._ram()
            if ram[self.lives] < lives:
                self.deaths[self._cell()] += 1
                return
            self._record(path, ram)
            if terminated or ram[self.won]:
                return

    def _play(self, action: int) -> bool:
        ended = False
        for buttons in self.actions[action]:
            _, _, terminated, truncated, _ = self.env.step(buttons)
            ended = ended or terminated or truncated
        return ended

    def _record(self, path: list[int], ram: np.ndarray) -> None:
        key = self._cell()
        if key not in self.archive or len(path) < len(self.archive[key].path):
            visits = self.archive[key].visits if key in self.archive else 0
            self.archive[key] = Cell(self.env.em.get_state(), list(path), visits)
        for address, value in zip(self.addresses, key, strict=True):
            self._milestone(f"{hex(address.location)}={value}", path)
        if ram[self.won]:
            self._milestone("won", path)

    def _milestone(self, name: str, path: list[int]) -> None:
        if name in self.milestones:
            return
        self.milestones[name] = {"decisions": len(path)}
        (self.out / f"{name}.state").write_bytes(self.env.em.get_state())
        (self.out / f"{name}.path.json").write_text(json.dumps(path))
        logger.info(f"Milestone {name} after {len(path)} decisions")

    def _cell(self) -> tuple[int, ...]:
        ram = self._ram()
        return tuple(int(ram[address.location]) // address.bucket for address in self.addresses)

    def _ram(self) -> np.ndarray:
        return self.env.get_ram()

    def _restore(self, emulator_state: bytes) -> None:
        self.env.em.set_state(emulator_state)
        self.env.data.reset()
        self.env.data.update_ram()


def main() -> None:
    arguments = parse_arguments()
    arguments.out.mkdir(parents=True, exist_ok=True)
    summary = Explorer(arguments).run(arguments.minutes * 60)
    (arguments.out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(json.dumps(summary))


if __name__ == "__main__":
    main()
