import math
import random
import time
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start

EXPLORE_DECISIONS = 24
KEEP_ACTION = 0.8
MAX_CELLS = 8000


@dataclass(slots=True, frozen=True)
class Trail:
    parent: "Trail | None"
    pressed: np.ndarray

    def frames(self) -> np.ndarray:
        chunks, trail = [], self
        while trail is not None:
            chunks.append(trail.pressed)
            trail = trail.parent
        return np.concatenate(chunks[::-1])


@dataclass(slots=True)
class Cell:
    start: Start
    trail: Trail | None
    frames: int
    visits: int


@dataclass(slots=True, frozen=True)
class Crossing:
    end: Start
    pressed: np.ndarray
    won: bool

    @property
    def frames(self) -> int:
        return len(self.pressed)


def choose(cells: list[Cell], rng: random.Random) -> Cell:
    cell = rng.choices(cells, weights=[1 / math.sqrt(1 + cell.visits) for cell in cells])[0]
    cell.visits += 1
    return cell


def search_segment(runner: SegmentRunner, start: Start, bound: int, seed: int, seconds: float) -> Crossing | None:
    rng, deadline = random.Random(seed), time.monotonic() + seconds
    archive: dict[CellKey, Cell] = {}
    runner.restore(start)
    archive[runner.key(0)] = Cell(start, None, 0, 0)
    best: Crossing | None = None
    while time.monotonic() < deadline:
        cell = choose(list(archive.values()), rng)
        runner.restore(cell.start)
        trail, frames, action = cell.trail, cell.frames, rng.randrange(len(runner.wrapper.action_table))
        for _ in range(EXPLORE_DECISIONS):
            if rng.random() > KEEP_ACTION:
                action = rng.randrange(len(runner.wrapper.action_table))
            outcome = runner.decide(action)
            trail, frames = Trail(trail, outcome.pressed), frames + len(outcome.pressed)
            if frames >= (bound if best is None else best.frames):
                break
            if outcome.won or (outcome.state != start.state and runner.forward(start.state, outcome.state)):
                best = Crossing(Start(outcome.state, runner.snapshot()), trail.frames(), outcome.won)
                break
            if not outcome.alive or outcome.state != start.state:
                break
            key = runner.key(frames)
            if key in archive and archive[key].frames <= frames:
                continue
            if key in archive or len(archive) < MAX_CELLS:
                archive[key] = Cell(Start(outcome.state, runner.snapshot()), trail, frames, 0)
    return best
