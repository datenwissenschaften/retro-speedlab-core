from dataclasses import dataclass

import msgspec
import numpy as np

from datenwissenschaften.reference.fetch import FRAMES_PER_SECOND
from datenwissenschaften.reference.frames import VideoFrames
from datenwissenschaften.reference.matcher import match_screens
from datenwissenschaften.reference.play import Span
from datenwissenschaften.reference.screens import Screenshot
from datenwissenschaften.reference.thumbnail import THUMB_SHAPE
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import Cell, Needs, cell_of


class Match(msgspec.Struct, frozen=True):
    seconds: float
    state: str
    level: int
    area: int
    x: int
    y: int
    cell: Cell
    distance: float


class Waypoint(msgspec.Struct, frozen=True):
    seconds: float
    cell: Cell
    state: str


class LevelMatches(msgspec.Struct, frozen=True):
    level: int
    frames: int
    matched: int
    waypoints: list[Waypoint]


class Matches(msgspec.Struct, frozen=True):
    levels: list[LevelMatches]
    matches: list[Match]


@dataclass(slots=True, frozen=True)
class LevelReading:
    summary: LevelMatches
    matches: list[Match]
    rows: list[list[int]]


def shot_cell(shot: Screenshot) -> Cell:
    return cell_of(Position(shot.area, shot.x, shot.y, 0, False))


def match_of(seconds: float, shot: Screenshot, distance: float) -> Match:
    return Match(seconds, shot.state, shot.level, shot.area, shot.x, shot.y, shot_cell(shot), distance)


def waypoints(matches: list[Match]) -> list[Waypoint]:
    changed = [match for index, match in enumerate(matches) if not index or match.cell != matches[index - 1].cell]
    return [Waypoint(match.seconds, match.cell, match.state) for match in changed]


def route_rows(matched: list[tuple[Match, Screenshot]], end: float, rate: float) -> list[list[int]]:
    fastest: dict[tuple[Cell, Needs], tuple[int, int]] = {}
    for match, shot in matched:
        key, left = (match.cell, tuple(shot.dynamics)), round((end - match.seconds) * rate)
        if key not in fastest or left < fastest[key][1]:
            fastest[key] = (round(match.seconds * rate), left)
    return [[*cell, frame, left, *needs] for (cell, needs), (frame, left) in sorted(fastest.items())]


def read_level(
    video: VideoFrames, playing: np.ndarray, span: Span, level: int, shots: list[Screenshot], rate: float
) -> LevelReading:
    start, end = span
    frames = np.arange(start, end)[playing[start:end]]
    placed = [shot for shot in shots if shot.level == level]
    emulator = np.asarray([shot.image() for shot in placed], dtype=np.uint8).reshape(-1, *THUMB_SHAPE)
    cells = np.asarray([shot_cell(shot) for shot in placed], dtype=np.int64).reshape(-1, 3)
    seconds = video.seconds[frames]
    matched = [
        (match_of(float(seconds[hit.frame]), placed[hit.screen], hit.distance), placed[hit.screen])
        for hit in match_screens(video.thumbs[frames], emulator, cells)
    ]
    matches = [match for match, _ in matched]
    summary = LevelMatches(level, len(frames), len(matches), waypoints(matches))
    return LevelReading(summary, matches, route_rows(matched, end / FRAMES_PER_SECOND, rate))
