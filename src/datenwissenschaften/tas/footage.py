from pathlib import Path
from typing import Any

import msgspec
import numpy as np

from datenwissenschaften.settings import RetroSpeedlabPaths
from datenwissenschaften.tas.journal import now
from datenwissenschaften.tas.movie import Segment, record_piece
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.report import Course
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.screen import Cut
from datenwissenschaften.tas.search import Crossing
from datenwissenschaften.tas.video import render_video, stack_videos

SCREEN_MOVIE = "screen.bk2"
MOVIE_MEMORY = "movie-ram.json"
PIECE_MOVIE = "piece.bk2"
COMPARE_PIECES = ("compare-before.bk2", "compare-after.bk2")
COMPARE_VIDEO = "compare.mp4"


def movie_field(first: str, frame_rate: float, segments: list[Segment]) -> dict[str, Any]:
    played, origin, cuts = 0, first, []
    for segment in segments:
        frames = len(segment.pressed)
        cuts.append({"state": origin, "reached": segment.state, "start_frame": played, "frames": frames})
        played, origin = played + frames, segment.state
    return {"frames": played, "seconds": played / frame_rate, "frame_rate": frame_rate, "segments": cuts}


def movie_cut(course: Course, segments: list[Segment], recording: Path, memory: dict[str, Any]) -> Cut:
    key = msgspec.msgpack.encode([segment.pressed for segment in segments])
    field = movie_field(course.first, course.frame_rate, segments)
    return Cut(key, (recording.read_bytes(),), field, {MOVIE_MEMORY: memory})


def compare_cut(runner: SegmentRunner, directory: Path, start: Start, before: Segment, after: Crossing) -> Cut:
    pieces = (np.asarray(before.pressed, dtype=np.int8), after.pressed)
    recordings = tuple(record_piece(runner, start, pressed, directory / PIECE_MOVIE) for pressed in pieces)
    field = {"state": after.end.state, "before_frames": len(before.pressed), "after_frames": after.frames, "at": now()}
    return Cut(b"".join(recordings), recordings, field, {})


def render_movie(paths: RetroSpeedlabPaths, directory: Path, recordings: tuple[bytes, ...]) -> Path:
    recording = directory / SCREEN_MOVIE
    atomic_write(recording, recordings[0])
    return render_video(paths, [recording])[0]


def render_compare(paths: RetroSpeedlabPaths, directory: Path, recordings: tuple[bytes, ...]) -> Path:
    pieces = [directory / name for name in COMPARE_PIECES]
    for piece, recording in zip(pieces, recordings, strict=True):
        atomic_write(piece, recording)
    before, after = render_video(paths, pieces)
    return stack_videos(before, after, directory / COMPARE_VIDEO)
