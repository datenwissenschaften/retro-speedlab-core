import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.recording import active_movie_path
from datenwissenschaften.tas.runner import SegmentRunner


@dataclass(slots=True, frozen=True)
class Segment:
    state: str
    pressed: list[list[int]]
    emulator_state: bytes
    won: bool


def verified_segments(runner: SegmentRunner, segments: list[Segment], movie: Path) -> int:
    emulator = runner.emulator
    emulator.auto_record(str(movie.parent))
    runner.power_on()
    recording = Path(active_movie_path(emulator))
    machine = runner.wrapper.state_machine
    verified = 0
    for segment in segments:
        for buttons in np.asarray(segment.pressed, dtype=np.int8):
            frame, *_ = runner.wrapper.env.step(buttons)
            machine.step(runner.wrapper.read_ram(), frame)
        if machine.state_name != segment.state and not (segment.won and machine.current_state._won()):
            break
        verified += 1
    emulator.stop_record()
    shutil.move(recording, movie)
    return verified
