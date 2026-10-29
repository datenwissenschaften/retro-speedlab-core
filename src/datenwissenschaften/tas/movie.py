import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.recording import active_movie_path
from datenwissenschaften.tas.runner import SegmentRunner
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.media import jpeg


@dataclass(slots=True, frozen=True)
class Segment:
    state: str
    pressed: list[list[int]]
    emulator_state: bytes
    won: bool


def verified_segments(runner: SegmentRunner, segments: list[Segment], movie: Path, shown: int) -> int:
    emulator = runner.emulator
    movie.parent.mkdir(parents=True, exist_ok=True)
    emulator.auto_record(str(movie.parent))
    origin = runner.power_on()
    recording = Path(active_movie_path(emulator))
    machine, frame_rate = runner.wrapper.state_machine, emulator.em.get_screen_rate()
    verified, played = 0, 0
    for index, segment in enumerate(segments):
        for buttons in np.asarray(segment.pressed, dtype=np.int8):
            frame, *_ = emulator.step(buttons)
            machine.step(runner.wrapper.read_ram())
            played += 1
            if index >= shown:
                status = {"step": played, "training_state": machine.state_name, "seconds": played / frame_rate}
                live_feed.record(jpeg(frame), status)
        if machine.state_name != segment.state and not (segment.won and machine.current_state._won()):
            break
        verified += 1
    emulator.stop_record()
    shutil.move(recording, movie)
    start = segments[shown - 1].state if 0 < shown <= len(segments) else origin.state
    result = {"start": start, "reached": machine.state_name, "seconds": played / frame_rate, "verified": verified}
    live_feed.finish_episode(len(segments), frame_rate, result)
    return verified
