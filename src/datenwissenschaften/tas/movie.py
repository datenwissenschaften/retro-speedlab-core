import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from datenwissenschaften.environment.recording import active_movie_path
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.tas.runner import SegmentRunner, Start

MEMORY_EVERY = 6


@dataclass(slots=True, frozen=True)
class Segment:
    state: str
    pressed: list[list[int]]
    emulator_state: bytes
    won: bool


@dataclass(slots=True, frozen=True)
class Replay:
    verified: int
    memory: dict[str, Any]
    ends: list[bytes]


def frames(segments: list[Segment]) -> int:
    return sum(len(segment.pressed) for segment in segments)


def memory_sample(ram: RamInfo) -> dict[str, int | None]:
    position = ram.position()
    if position is None:
        return {**ram.describe(), "area": None, "x": None, "y": None, "height": None}
    return {**ram.describe(), "area": position.area, "x": position.x, "y": position.y, "height": position.height}


def memory_field(samples: list[dict[str, int | None]]) -> dict[str, Any]:
    keys = list(samples[0]) if samples else []
    for sample in samples:
        if list(sample) != keys:
            raise ValueError(f"The game memory names {list(sample)} instead of {keys} during the movie")
    return {"every": MEMORY_EVERY, "keys": keys, "samples": [list(sample.values()) for sample in samples]}


@contextmanager
def recording(runner: SegmentRunner, movie: Path) -> Iterator[None]:
    emulator = runner.emulator
    movie.parent.mkdir(parents=True, exist_ok=True)
    emulator.auto_record(str(movie.parent))
    runner.power_on()
    recorded = Path(active_movie_path(emulator))
    yield
    emulator.stop_record()
    shutil.move(recorded, movie)


def record_piece(runner: SegmentRunner, start: Start, pressed: np.ndarray, movie: Path) -> bytes:
    with recording(runner, movie):
        runner.restore(start)
        for buttons in pressed:
            runner.emulator.step(buttons)
    return movie.read_bytes()


def verified_segments(runner: SegmentRunner, segments: list[Segment], movie: Path) -> Replay:
    machine = runner.wrapper.state_machine
    verified, played, samples, ends = 0, 0, [], []
    with recording(runner, movie):
        for segment in segments:
            for buttons in np.asarray(segment.pressed, dtype=np.int8):
                runner.emulator.step(buttons)
                ram = runner.wrapper.read_ram()
                machine.step(ram)
                if played % MEMORY_EVERY == 0:
                    samples.append(memory_sample(ram))
                played += 1
            if machine.state_name != segment.state and not (segment.won and machine.current_state._won()):
                break
            verified += 1
            ends.append(runner.snapshot())
    return Replay(verified, memory_field(samples), ends)
