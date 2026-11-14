from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
from speedlab_engine.movie import Movie, encode

from datenwissenschaften.environment.wrapper import known_lives, lost_life
from datenwissenschaften.ram import RamInfo, viewed
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import left_at_once

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
        located: dict[str, int | None] = {"area": None, "x": None, "y": None, "height": None}
    else:
        located = {"area": position.area, "x": position.x, "y": position.y, "height": position.height}
    return viewed(type(ram).shown_fields(), {**ram.describe(), **located})


def memory_field(samples: list[dict[str, int | None]]) -> dict[str, Any]:
    keys = list(samples[0]) if samples else []
    for sample in samples:
        if list(sample) != keys:
            raise ValueError(f"The game memory names {list(sample)} instead of {keys} during the movie")
    return {"every": MEMORY_EVERY, "keys": keys, "samples": [list(sample.values()) for sample in samples]}


def recorded(runner: SegmentRunner, start: Start, pressed: list[list[int]]) -> bytes:
    emulator = runner.emulator
    masks = [emulator.action_to_array(np.asarray(buttons, dtype=np.int8))[0] for buttons in pressed]
    frames = encode(np.asarray(masks, dtype=np.uint8))
    return msgspec.msgpack.encode(Movie(emulator.gamename, emulator.version, start.emulator_state, frames))


def pieces_pressed(segments: list[Segment]) -> list[list[int]]:
    return [buttons for segment in segments for buttons in segment.pressed]


def verified_segments(runner: SegmentRunner, start: Start, segments: list[Segment]) -> Replay:
    machine = runner.wrapper.state_machine
    verified, played = 0, 0
    samples: list[dict[str, int | None]] = []
    ends: list[bytes] = []
    runner.restore(start)
    lives, died = runner.lives, False
    for segment in segments:
        entered = machine.state_name
        for buttons in np.asarray(segment.pressed, dtype=np.int8):
            runner.emulator.step(buttons)
            ram = runner.wrapper.read_ram()
            _, terminated, _ = machine.step(ram)
            died = died or terminated or lost_life(lives, ram)
            lives = known_lives(lives, ram)
            if played % MEMORY_EVERY == 0:
                samples.append(memory_sample(ram))
            played += 1
        if died or (machine.state_name != segment.state and not (segment.won and machine.current_state.won())):
            break
        if left_at_once(segment.pressed) and (segment.won or segment.state != entered):
            break
        verified += 1
        ends.append(runner.snapshot())
    return Replay(verified, memory_field(samples), ends)
