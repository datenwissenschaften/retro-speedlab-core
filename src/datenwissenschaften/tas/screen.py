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

Render = Callable[[tuple[bytes, ...]], Path]


@dataclass(slots=True, frozen=True)
class Cut:
    key: bytes
    recordings: tuple[bytes, ...]
    field: dict[str, Any]
    attachments: dict[str, dict[str, Any]]


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
            self._present(cut)
            with self.changed:
                self.busy = False
                self.changed.notify_all()

    def _present(self, cut: Cut) -> None:
        began = time.monotonic()
        try:
            video = self.render(cut.recordings)
        except subprocess.CalledProcessError as error:
            logger.error(f"The {self.name} video does not render: {error.stderr}")
            return
        field = {"version": version(video), **cut.field}
        logger.info(f"Rendered the {self.name} video {field['version']} in {time.monotonic() - began:.1f} s")
        while self.pending is None:
            if self._uploaded(video, field["version"], cut):
                self.show(field)
                return
            with self.changed:
                self.changed.wait_for(lambda: self.pending is not None, RETRY_SECONDS)

    def _uploaded(self, video: Path, shown: str, cut: Cut) -> bool:
        if self.upload is None:
            return True
        try:
            self.upload.put(video, shown, cut.attachments)
        except httpx.HTTPError as error:
            logger.warning(
                f"The {self.name} video does not reach the backend, retrying in {RETRY_SECONDS:.0f} s: {error}"
            )
            return False
        return True
