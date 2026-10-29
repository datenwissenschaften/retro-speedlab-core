from dataclasses import dataclass
from pathlib import Path

import msgspec

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.report import Course
from datenwissenschaften.tas.runner import Start

ROUTES_FILE = "levels.msgpack"
FULL_RUN = "Full run"


@dataclass(slots=True, frozen=True)
class Part:
    name: str
    entry_state: str
    exit: str | None


@dataclass(slots=True, frozen=True)
class Leg:
    name: str
    entry: Start
    exit: str | None


def parts_of(course: Course) -> list[Part]:
    entries = [states[0] for states in course.levels.values()]
    exits = [*entries, None]
    opening = Part(course.first, course.first, exits[0])
    return [opening, *(Part(level, entries[index], exits[index + 1]) for index, level in enumerate(course.levels))]


class Route(msgspec.Struct, frozen=True):
    entry: bytes
    segments: list[Segment]


def finished(segments: list[Segment], until: str | None) -> bool:
    return bool(segments) and (segments[-1].won or segments[-1].state == until)


def load_progress(path: Path) -> list[Segment]:
    return msgspec.msgpack.decode(path.read_bytes(), type=list[Segment]) if path.is_file() else []


def save_progress(path: Path, segments: list[Segment]) -> None:
    atomic_write(path, msgspec.msgpack.encode(segments))


class Routes:
    def __init__(self, path: Path, progress: Path, parts: list[Part]) -> None:
        self.path = path
        self.progress = progress
        self.parts = parts
        self.routes = msgspec.msgpack.decode(path.read_bytes(), type=dict[str, Route]) if path.is_file() else {}

    def segments(self, name: str) -> list[Segment]:
        return list(self.routes[name].segments) if name in self.routes else []

    def entry(self, part: Part, power_on: Start) -> Start | None:
        if part.name in self.routes:
            return Start(part.entry_state, self.routes[part.name].entry)
        if part is self.parts[0]:
            return power_on
        before = self.parts[self.parts.index(part) - 1]
        if not finished(self.segments(before.name), before.exit):
            return None
        return Start(part.entry_state, self.routes[before.name].segments[-1].emulator_state)

    def save(self, leg: Leg, segments: list[Segment]) -> None:
        if leg.name == FULL_RUN:
            save_progress(self.progress, segments)
            return
        self.routes = {**self.routes, leg.name: Route(leg.entry.emulator_state, segments)}
        atomic_write(self.path, msgspec.msgpack.encode(self.routes))

    def view(self, name: str, segments: list[Segment]) -> list[Segment]:
        if name == FULL_RUN:
            return segments
        stitched = []
        for part in self.parts:
            route = segments if part.name == name else self.segments(part.name)
            stitched += route
            if not finished(route, part.exit):
                return stitched
        return stitched

    def stitched(self) -> list[Segment]:
        first = self.parts[0].name
        return self.view(first, self.segments(first))

    def split(self, segments: list[Segment], power_on: Start) -> None:
        entry, remaining = power_on.emulator_state, list(segments)
        for part in self.parts:
            taken = next((index + 1 for index, piece in enumerate(remaining) if piece.state == part.exit), None)
            route, remaining = (remaining, []) if taken is None else (remaining[:taken], remaining[taken:])
            self.save(Leg(part.name, Start(part.entry_state, entry), part.exit), route)
            if taken is None:
                return
            entry = route[-1].emulator_state
