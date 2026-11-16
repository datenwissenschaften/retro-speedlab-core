import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec
from speedlab_engine.video import Encoder

from datenwissenschaften.settings import RetroSpeedlabPaths
from datenwissenschaften.speedrun.journal import now
from datenwissenschaften.speedrun.movie import Segment, recorded
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.report import Course
from datenwissenschaften.speedrun.routes import FULL_RUN, Leg
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.screen import Cut
from datenwissenschaften.speedrun.search import Crossing
from datenwissenschaften.speedrun.video import render_video, stack_videos

MOVIE = "movie"
COMPARE = "compare"
RUN = "run"
SCREEN_MOVIE = "screen.movie"
MOVIE_MEMORY = "movie-ram.json"
COMPARE_PIECES = ("compare-before.movie", "compare-after.movie")
COMPARE_VIDEO = "compare.mp4"
RUN_PIECES = "run.msgpack"
RUN_VIDEO = "run.mp4"
RUN_VIEW = "datenwissenschaften.speedrun.run_view"
LOW_PRIORITY = ("nice", "-n", "19")


def movie_field(first: str, frame_rate: float, segments: list[Segment]) -> dict[str, Any]:
    played, origin = 0, first
    cuts: list[dict[str, Any]] = []
    for segment in segments:
        frames = len(segment.pressed)
        cuts.append({"state": origin, "reached": segment.state, "start_frame": played, "frames": frames})
        played, origin = played + frames, segment.state
    return {"frames": played, "seconds": played / frame_rate, "frame_rate": frame_rate, "segments": cuts}


def movie_cut(course: Course, leg: Leg, segments: list[Segment], recording: bytes, memory: dict[str, Any]) -> Cut:
    key = msgspec.msgpack.encode([leg.name, *(segment.pressed for segment in segments)])
    field = {**movie_field(leg.entry.state, course.frame_rate, segments), "part": leg.name}
    return Cut(key, (recording,), field, {MOVIE_MEMORY: memory})


def run_cut(course: Course, segments: list[Segment]) -> Cut:
    key = msgspec.msgpack.encode([segment.pressed for segment in segments])
    field = {**movie_field(course.first, course.frame_rate, segments), "part": FULL_RUN}
    return Cut(key, (msgspec.msgpack.encode(segments),), field, {})


def compare_cut(runner: SegmentRunner, start: Start, before: Segment, after: Crossing) -> Cut:
    recordings = (recorded(runner, start, before.pressed), recorded(runner, start, after.pressed.tolist()))
    field = {"state": after.end.state, "before_frames": len(before.pressed), "after_frames": after.frames, "at": now()}
    return Cut(b"".join(recordings), recordings, field, {})


def render_movie(paths: RetroSpeedlabPaths, encoder: Encoder, directory: Path, recordings: tuple[bytes, ...]) -> Path:
    recording = directory / SCREEN_MOVIE
    atomic_write(recording, recordings[0])
    return render_video(paths, encoder, [recording])[0]


def render_compare(paths: RetroSpeedlabPaths, encoder: Encoder, directory: Path, recordings: tuple[bytes, ...]) -> Path:
    pieces = [directory / name for name in COMPARE_PIECES]
    for piece, recording in zip(pieces, recordings, strict=True):
        atomic_write(piece, recording)
    before, after = render_video(paths, encoder, pieces)
    return stack_videos(before, after, directory / COMPARE_VIDEO, encoder)


def render_run(wrapper: str, config: Path, directory: Path, recordings: tuple[bytes, ...]) -> Path:
    pieces, video = directory / RUN_PIECES, directory / RUN_VIDEO
    atomic_write(pieces, recordings[0])
    command = [*LOW_PRIORITY, sys.executable, "-m", RUN_VIEW, wrapper, str(config), str(pieces), str(video)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return video
