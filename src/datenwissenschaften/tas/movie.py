import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.recording import active_movie_path
from datenwissenschaften.tas.runner import SegmentRunner, Spot, spot


@dataclass(slots=True, frozen=True)
class Segment:
    state: str
    pressed: list[list[int]]
    emulator_state: bytes
    won: bool


@dataclass(slots=True, frozen=True)
class Replay:
    verified: int
    route: list[Spot]


def frames(segments: list[Segment]) -> int:
    return sum(len(segment.pressed) for segment in segments)


def verified_segments(runner: SegmentRunner, segments: list[Segment], movie: Path) -> Replay:
    emulator = runner.emulator
    movie.parent.mkdir(parents=True, exist_ok=True)
    emulator.auto_record(str(movie.parent))
    runner.power_on()
    recording = Path(active_movie_path(emulator))
    machine = runner.wrapper.state_machine
    verified, route = 0, []
    for segment in segments:
        for buttons in np.asarray(segment.pressed, dtype=np.int8):
            emulator.step(buttons)
            ram = runner.wrapper.read_ram()
            machine.step(ram)
            if (position := ram.position()) is not None and (not route or route[-1] != spot(position)):
                route.append(spot(position))
        if machine.state_name != segment.state and not (segment.won and machine.current_state._won()):
            break
        verified += 1
    emulator.stop_record()
    shutil.move(recording, movie)
    return Replay(verified, route)
