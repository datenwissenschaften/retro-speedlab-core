import hashlib
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import msgspec
from loguru import logger

from datenwissenschaften.settings import RetroSpeedlabPaths
from datenwissenschaften.tas.movie import Segment, frames
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.report import Course
from datenwissenschaften.tas.video import render_video
from datenwissenschaften.ui.movie_upload import MovieUpload

SCREEN_MOVIE = "screen.bk2"
VERSION_LENGTH = 12
RETRY_SECONDS = 30.0


@dataclass(slots=True, frozen=True)
class Cut:
    recording: bytes
    movie: dict[str, Any]


def movie_field(first: str, frame_rate: float, segments: list[Segment]) -> dict[str, Any]:
    played, origin, cuts = 0, first, []
    for segment in segments:
        cuts.append({"state": origin, "start_frame": played, "frames": len(segment.pressed)})
        played, origin = played + len(segment.pressed), segment.state
    return {"frames": played, "seconds": played / frame_rate, "frame_rate": frame_rate, "segments": cuts}


def version(video: Path) -> str:
    return hashlib.sha256(video.read_bytes()).hexdigest()[:VERSION_LENGTH]


class MovieScreen:
    def __init__(
        self,
        paths: RetroSpeedlabPaths,
        directory: Path,
        course: Course,
        upload: MovieUpload | None,
        show: Callable[[dict[str, Any]], None],
    ) -> None:
        self.paths = paths
        self.recording = directory / SCREEN_MOVIE
        self.course = course
        self.upload = upload
        self.show = show
        self.changed = threading.Condition()
        self.pending: Cut | None = None
        self.busy = False
        self.queued: bytes | None = None

    def start(self) -> None:
        threading.Thread(target=self._work, name="movie-screen", daemon=True).start()

    def screen(self, segments: list[Segment], recording: Path) -> None:
        pressed = msgspec.msgpack.encode([segment.pressed for segment in segments])
        if not frames(segments) or pressed == self.queued:
            return
        self.queued = pressed
        with self.changed:
            self.pending = Cut(recording.read_bytes(), movie_field(self.course.first, self.course.frame_rate, segments))
            self.changed.notify_all()

    def settle(self) -> Path:
        with self.changed:
            self.changed.wait_for(lambda: self.pending is None and not self.busy)
        return self.recording.with_suffix(".mp4")

    def _work(self) -> None:
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.pending is not None)
                cut, self.pending, self.busy = self.pending, None, True
            self._present(cut)
            with self.changed:
                self.busy = False
                self.changed.notify_all()

    def _present(self, cut: Cut) -> None:
        atomic_write(self.recording, cut.recording)
        began = time.monotonic()
        try:
            video = render_video(self.paths, self.recording)
        except subprocess.CalledProcessError as error:
            logger.error(f"The movie of {cut.movie['frames']} frames does not render: {error.stderr}")
            return
        movie = {"version": version(video), **cut.movie}
        logger.info(f"Rendered movie {movie['version']}: {movie['frames']} frames in {time.monotonic() - began:.1f} s")
        while self.pending is None:
            if self._uploaded(video):
                self.show(movie)
                return
            with self.changed:
                self.changed.wait_for(lambda: self.pending is not None, RETRY_SECONDS)

    def _uploaded(self, video: Path) -> bool:
        if self.upload is None:
            return True
        try:
            self.upload.put(video)
        except httpx.HTTPError as error:
            logger.warning(f"The movie does not reach the backend, retrying in {RETRY_SECONDS:.0f} s: {error}")
            return False
        return True
