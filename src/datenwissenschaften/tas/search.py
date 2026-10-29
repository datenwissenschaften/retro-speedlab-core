import math
import random
import sys
import time
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start

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
CHECKPOINT_GAIN = 1.0


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


@dataclass(slots=True, frozen=True)
class Checkpoint:
    end: Start
    pressed: np.ndarray
    gain: float

    @property
    def frames(self) -> int:
        return len(self.pressed)


@dataclass(slots=True, frozen=True)
class Searched:
    crossings: list[Crossing]
    checkpoint: Checkpoint | None


def farthest(checkpoints: list[Checkpoint | None]) -> Checkpoint | None:
    found = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    return max(found, key=lambda checkpoint: (checkpoint.gain, -checkpoint.frames), default=None)


def checkpoint_of(archive: dict[CellKey, Cell]) -> Checkpoint | None:
    reached = [cell for cell in archive.values() if cell.trail is not None and cell.score >= CHECKPOINT_GAIN]
    if not reached:
        return None
    cell = max(reached, key=lambda cell: (cell.score, -cell.frames))
    return Checkpoint(cell.start, cell.trail.frames(), cell.score)


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


def confirmed(runner: SegmentRunner, pressed: np.ndarray, won: bool) -> Crossing | None:
    state = runner.wrapper.state_machine.state_name
    crossing = Crossing(Start(state, runner.snapshot()), pressed, won, runner.key(0))
    return crossing if won or holds(runner, state, np.zeros_like(pressed[-1])) else None


def search_segment(
    runner: SegmentRunner, start: Start, bound: int, seed: int, seconds: float, live: WorkerLive | None
) -> Searched:
    rng, deadline = random.Random(seed), time.monotonic() + seconds
    actions = len(runner.wrapper.action_table)
    runner.restore(start)
    archive = {runner.key(0): Cell(start, None, 0, 0, runner.remaining(), 0.0)}
    crossings: list[Crossing] = []
    while time.monotonic() < deadline:
        cell = choose(list(archive.values()), rng)
        runner.restore(cell.start)
        trail, frames, score, action = cell.trail, cell.frames, cell.score, rng.randrange(actions)
        last, decisions, still = runner.key(frames), 0, 0
        while decisions < EXPLORE_DECISIONS and still < STILL_FRAMES:
            if rng.random() > KEEP_ACTION:
                action = rng.randrange(actions)
            outcome = runner.decide(action)
            trail, frames, score = Trail(trail, outcome.pressed), frames + len(outcome.pressed), score + outcome.reward
            if live is not None:
                live.step(runner, archive, crossings, len(outcome.pressed))
            if frames >= (bound if len(crossings) < BEAM else crossings[-1].frames):
                break
            if leaves(runner, start, outcome.state, outcome.won):
                if (crossing := confirmed(runner, trail.frames(), outcome.won)) is not None:
                    crossings = fastest_distinct([*crossings, crossing])
                break
            if not outcome.alive or outcome.state != start.state:
                break
            key = runner.key(frames)
            if key == last:
                still += len(outcome.pressed)
                continue
            last, decisions, still = key, decisions + 1, 0
            if key in archive and archive[key].frames <= frames:
                continue
            if key not in archive and len(archive) >= MAX_CELLS:
                del archive[min(archive, key=lambda kept: (archive[kept].score, -archive[kept].visits))]
            archive[key] = Cell(Start(outcome.state, runner.snapshot()), trail, frames, 0, runner.remaining(), score)
    return Searched(crossings, checkpoint_of(archive))


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
