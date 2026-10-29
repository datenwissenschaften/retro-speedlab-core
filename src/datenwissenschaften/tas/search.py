import math
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start

EXPLORE_DECISIONS = 24
KEEP_ACTION = 0.8
MAX_CELLS = 8000
BEAM = 3
ROUTE_GAIN = 50.0
SHAVE_WINDOWS = (8, 4, 2, 1)
EXPLOIT = 0.5
LEADING_SHARE = 10
UNBOUNDED = sys.maxsize


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
    remaining: int | None
    score: float

    @property
    def weight(self) -> float:
        guide = 1.0 if self.remaining is None else 1.0 + ROUTE_GAIN / (1 + self.remaining)
        return guide / math.sqrt(1 + self.visits)


@dataclass(slots=True, frozen=True)
class Crossing:
    end: Start
    pressed: np.ndarray
    won: bool
    key: CellKey

    @property
    def frames(self) -> int:
        return len(self.pressed)


def fastest_distinct(crossings: list[Crossing]) -> list[Crossing]:
    fastest: dict[CellKey, Crossing] = {}
    for crossing in sorted(crossings, key=lambda crossing: crossing.frames):
        fastest.setdefault(crossing.key, crossing)
    return list(fastest.values())[:BEAM]


def choose(cells: list[Cell], rng: random.Random) -> Cell:
    if rng.random() < EXPLOIT:
        cells = sorted(cells, key=lambda cell: cell.score)[-max(1, len(cells) // LEADING_SHARE) :]
    cell = rng.choices(cells, weights=[cell.weight for cell in cells])[0]
    cell.visits += 1
    return cell


def leaves(runner: SegmentRunner, start: Start, state: str, won: bool) -> bool:
    return won or (state != start.state and runner.forward(start.state, state))


def search_segment(runner: SegmentRunner, start: Start, bound: int, seed: int, seconds: float) -> list[Crossing]:
    rng, deadline = random.Random(seed), time.monotonic() + seconds
    actions = len(runner.wrapper.action_table)
    runner.restore(start)
    archive = {runner.key(0): Cell(start, None, 0, 0, runner.remaining(), 0.0)}
    crossings: list[Crossing] = []
    while time.monotonic() < deadline:
        cell = choose(list(archive.values()), rng)
        runner.restore(cell.start)
        trail, frames, score, action = cell.trail, cell.frames, cell.score, rng.randrange(actions)
        for _ in range(EXPLORE_DECISIONS):
            if rng.random() > KEEP_ACTION:
                action = rng.randrange(actions)
            outcome = runner.decide(action)
            trail, frames, score = Trail(trail, outcome.pressed), frames + len(outcome.pressed), score + outcome.reward
            if frames >= (bound if len(crossings) < BEAM else crossings[-1].frames):
                break
            if leaves(runner, start, outcome.state, outcome.won):
                ending = Start(outcome.state, runner.snapshot())
                crossings = fastest_distinct([*crossings, Crossing(ending, trail.frames(), outcome.won, runner.key(0))])
                break
            if not outcome.alive or outcome.state != start.state:
                break
            key = runner.key(frames)
            if key in archive and archive[key].frames <= frames:
                continue
            if key not in archive and len(archive) >= MAX_CELLS:
                del archive[min(archive, key=lambda kept: (archive[kept].score, -archive[kept].visits))]
            archive[key] = Cell(Start(outcome.state, runner.snapshot()), trail, frames, 0, runner.remaining(), score)
    return crossings


def replay(runner: SegmentRunner, start: Start, pressed: np.ndarray) -> Crossing | None:
    runner.restore(start)
    machine = runner.wrapper.state_machine
    for frame, buttons in enumerate(pressed, 1):
        alive, won, _ = runner.press(buttons)
        if leaves(runner, start, machine.state_name, won):
            return Crossing(Start(machine.state_name, runner.snapshot()), pressed[:frame], won, runner.key(0))
        if not alive or machine.state_name != start.state:
            return None
    return None


def shave(runner: SegmentRunner, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    deadline = time.monotonic() + seconds
    for window in SHAVE_WINDOWS:
        position = 0
        while position < crossing.frames and time.monotonic() < deadline:
            shorter = replay(runner, start, np.delete(crossing.pressed, slice(position, position + window), axis=0))
            if shorter is not None and shorter.frames < crossing.frames and shorter.end.state == crossing.end.state:
                crossing = shorter
            else:
                position += window
    return crossing
