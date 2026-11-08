import math
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.speedrun.runner import CellKey, SegmentRunner, Start

EXPLORE_DECISIONS = 24
STILL_FRAMES = 1800
HOLD_FRAMES = 240
KEEP_ACTION = 0.8
MAX_CELLS = 8000
BEAM = 3
ROUTE_GAIN = 50.0
SHAVE_WINDOWS = (8, 4, 2, 1)
EXPLOIT = 0.5
LEADING_SHARE = 10
UNBOUNDED = sys.maxsize
CHECKPOINT_GAIN = 10.0
ROUTE_CHECKPOINT = 60
NOVELTY_GAIN = 2.0
NOVELTY_HALF = 8
FIRST_FRAME = 1


@dataclass(slots=True, frozen=True)
class Trail:
    parent: "Trail | None"
    pressed: np.ndarray

    def frames(self) -> np.ndarray:
        chunks: list[np.ndarray] = []
        trail: Trail | None = self
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
    novelty: int

    @property
    def weight(self) -> float:
        guide = 1.0 if self.remaining is None else 1.0 + ROUTE_GAIN / (1 + self.remaining)
        fresh = 1.0 + NOVELTY_GAIN * self.novelty / (self.novelty + NOVELTY_HALF)
        return guide * fresh / math.sqrt(1 + self.visits)


@dataclass(slots=True, frozen=True)
class Crossing:
    end: Start
    pressed: np.ndarray
    won: bool
    key: CellKey

    @property
    def frames(self) -> int:
        return len(self.pressed)


@dataclass(slots=True, frozen=True)
class Checkpoint:
    end: Start
    pressed: np.ndarray
    gain: float
    remaining: int | None

    @property
    def frames(self) -> int:
        return len(self.pressed)

    @property
    def guided(self) -> bool:
        return self.remaining is not None


@dataclass(slots=True, frozen=True)
class Searched:
    crossings: list[Crossing]
    checkpoint: Checkpoint | None


def progress(checkpoint: Checkpoint) -> tuple[int, float, int]:
    if checkpoint.remaining is None:
        return 0, checkpoint.gain, -checkpoint.frames
    return 1, -checkpoint.remaining, -checkpoint.frames


def farthest(checkpoints: list[Checkpoint | None]) -> Checkpoint | None:
    found = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    return max(found, key=progress, default=None)


def on_route(archive: dict[CellKey, Cell]) -> Cell | None:
    starts = [cell.remaining for cell in archive.values() if cell.trail is None and cell.remaining is not None]
    cells = archive.values()
    mapped = [(cell.remaining, cell) for cell in cells if cell.trail is not None and cell.remaining is not None]
    limit = starts[0] - ROUTE_CHECKPOINT if starts else None
    closer = [(left, cell) for left, cell in mapped if limit is None or left <= limit]
    best = min(closer, key=lambda pair: (pair[0], pair[1].frames), default=None)
    return None if best is None else best[1]


def by_reward(archive: dict[CellKey, Cell]) -> Cell | None:
    reached = [cell for cell in archive.values() if cell.trail is not None and cell.score >= CHECKPOINT_GAIN]
    return max(reached, key=lambda cell: (cell.score, -cell.frames), default=None)


def checkpoint_of(archive: dict[CellKey, Cell]) -> Checkpoint | None:
    guided = on_route(archive)
    cell = guided or by_reward(archive)
    if cell is None:
        return None
    if cell.trail is None:
        raise RuntimeError("The checkpoint cell keeps no trail")
    return Checkpoint(cell.start, cell.trail.frames(), cell.score, None if guided is None else cell.remaining)


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


def holds(runner: SegmentRunner, state: str, idle: np.ndarray) -> bool:
    machine = runner.wrapper.state_machine
    for _ in range(HOLD_FRAMES):
        alive, won, _ = runner.press(idle)
        if won:
            return True
        if not alive or runner.forward(machine.state_name, state):
            return False
    return True


def left_at_once(pressed: list[list[int]] | np.ndarray) -> bool:
    return len(pressed) <= FIRST_FRAME


def confirmed(runner: SegmentRunner, pressed: np.ndarray, won: bool) -> Crossing | None:
    if left_at_once(pressed):
        return None
    state = runner.wrapper.state_machine.state_name
    crossing = Crossing(Start(state, runner.snapshot()), pressed, won, runner.key(0))
    return crossing if won or holds(runner, state, np.zeros_like(pressed[-1])) else None


def replay(runner: SegmentRunner, start: Start, pressed: np.ndarray) -> Crossing | None:
    runner.restore(start)
    machine = runner.wrapper.state_machine
    for frame, buttons in enumerate(pressed, 1):
        alive, won, _ = runner.press(buttons)
        if leaves(runner, start, machine.state_name, won):
            return confirmed(runner, pressed[:frame], won)
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
