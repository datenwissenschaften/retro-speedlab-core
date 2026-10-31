from pathlib import Path

import msgspec

from datenwissenschaften.tas.movie import Segment, frames
from datenwissenschaften.tas.paths import atomic_write

SHOWN_FILE = "shown.json"


class Shown(msgspec.Struct, frozen=True):
    states: list[str]
    pieces: int
    frames: int


def shown_of(states: tuple[str, ...], entry: str, segments: list[Segment]) -> Shown:
    route = [entry, *(segment.state for segment in segments)]
    pieces = sum(before != after for before, after in zip(route, route[1:], strict=False))
    return Shown(list(states), pieces, frames(segments))


def worth_showing(path: Path, states: tuple[str, ...], entry: str, segments: list[Segment]) -> bool:
    candidate = shown_of(states, entry, segments)
    if path.is_file():
        shown = msgspec.json.decode(path.read_bytes(), type=Shown)
        if shown.states == candidate.states and (candidate.pieces, -candidate.frames) < (shown.pieces, -shown.frames):
            return False
    atomic_write(path, msgspec.json.encode(candidate))
    return True
