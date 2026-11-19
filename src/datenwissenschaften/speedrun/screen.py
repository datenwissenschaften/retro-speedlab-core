import hashlib
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from datenwissenschaften.ui.video_upload import VideoUpload

VERSION_LENGTH = 12
RETRY_SECONDS = 30.0
LONGEST_RETRY_SECONDS = 30 * 60.0

Render = Callable[[tuple[bytes, ...]], Path]


@dataclass(slots=True, frozen=True)
class Cut:
    key: bytes
    recordings: tuple[bytes, ...]
    field: dict[str, Any]
    attachments: dict[str, dict[str, Any]]


def retry_seconds(failures: int) -> float:
    return min(RETRY_SECONDS * 2**failures, LONGEST_RETRY_SECONDS)


def version(video: Path) -> str:
    return hashlib.sha256(video.read_bytes()).hexdigest()[:VERSION_LENGTH]


class Screen:
    def __init__(
        self, name: str, render: Render, upload: VideoUpload | None, show: Callable[[dict[str, Any]], None]
    ) -> None:
        self.name = name
        self.render = render
        self.upload = upload
        self.show = show
        self.changed = threading.Condition()
        self.pending: Cut | None = None
        self.busy = False
        self.queued: bytes | None = None

    def start(self) -> None:
        threading.Thread(target=self._work, name=f"{self.name}-screen", daemon=True).start()

    def queue(self, cut: Cut) -> None:
        if cut.key == self.queued:
            return
        self.queued = cut.key
        with self.changed:
            self.pending = cut
            self.changed.notify_all()

    def settle(self) -> None:
        with self.changed:
            self.changed.wait_for(lambda: self.pending is None and not self.busy)

    def _work(self) -> None:
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.pending is not None)
                cut, self.pending, self.busy = self.pending, None, True
            if cut is None:
                raise RuntimeError("The screen woke without a cut to show")
            self._present(cut)
            with self.changed:
                self.busy = False
                self.changed.notify_all()

    def _present(self, cut: Cut) -> None:
        began, failures = time.monotonic(), 0
        while (video := self._rendered(cut, retry_seconds(failures))) is None:
            if self._superseded(retry_seconds(failures)):
                return
            failures += 1
        field = {"version": version(video), **cut.field}
        logger.info(f"Rendered the {self.name} video {field['version']} in {time.monotonic() - began:.1f} s")
        failures = 0
        while not self._uploaded(video, field["version"], cut, retry_seconds(failures)):
            if self._superseded(retry_seconds(failures)):
                return
            failures += 1
        self.show(field)

    def _superseded(self, seconds: float) -> bool:
        with self.changed:
            return self.changed.wait_for(lambda: self.pending is not None, seconds)

    def _rendered(self, cut: Cut, retry: float) -> Path | None:
        try:
            return self.render(cut.recordings)
        except subprocess.CalledProcessError as error:
            logger.error(f"The {self.name} video does not render, retrying in {retry:.0f} s: {error.stderr}")
            return None

    def _uploaded(self, video: Path, shown: str, cut: Cut, retry: float) -> bool:
        if self.upload is None:
            return True
        try:
            self.upload.put(video, shown, cut.attachments)
        except httpx.HTTPError as error:
            logger.warning(f"The {self.name} video does not reach the backend, retrying in {retry:.0f} s: {error}")
            return False
        return True
