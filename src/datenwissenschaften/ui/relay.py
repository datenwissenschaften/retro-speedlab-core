import gzip
import json
import threading
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from datenwissenschaften.settings import UISettings, UploadSettings
from datenwissenschaften.ui.telemetry import metadata

RELAY_SECONDS = 3.0
ATTEMPT_SECONDS = 0.5
ATTEMPT_BYTES = 256 * 1024
TIMEOUT_SECONDS = 120.0


def _datenwissenschaften_version() -> str:
    try:
        return version("datenwissenschaften")
    except PackageNotFoundError:
        return "DEVELOPMENT"


DATENWISSENSCHAFTEN_VERSION = _datenwissenschaften_version()


def relay_key(upload: UploadSettings) -> str:
    if upload.api_key is None:
        raise RuntimeError("ui.enable relays the stream to the backend and needs upload.api_key.")
    return upload.api_key


def backend_client(url: str, api_key: str) -> httpx.Client:
    return httpx.Client(base_url=f"{url}/live", headers={"X-API-Key": api_key}, timeout=TIMEOUT_SECONDS)


class LiveRelay:
    def __init__(self, url: str, api_key: str, ui: UISettings, attempt: Path) -> None:
        self.url = url
        self.api_key = api_key
        self.ui = ui
        self.attempt = attempt
        self.attempt_shown: int | None = None
        self.stopped = threading.Event()

    def start(self) -> None:
        pushes = ((self._push_snapshot, RELAY_SECONDS), (self._push_attempt, ATTEMPT_SECONDS))
        for push, seconds in pushes:
            name = f"relay{push.__name__}"
            threading.Thread(target=self._repeat, args=(push, seconds), name=name, daemon=True).start()

    def stop(self) -> None:
        self.stopped.set()

    def _repeat(self, push: Callable[[httpx.Client], None], seconds: float) -> None:
        failing = False
        with backend_client(self.url, self.api_key) as client:
            while not self.stopped.is_set():
                try:
                    push(client)
                    if failing:
                        logger.info("The live relay reaches the backend again")
                    failing = False
                except httpx.HTTPError as error:
                    if not failing:
                        logger.warning(f"The live relay cannot reach the backend: {error}")
                    failing = True
                self.stopped.wait(seconds)

    def _push_snapshot(self, client: httpx.Client) -> None:
        _put_document(client, "/snapshot", stream_snapshot(self.ui))

    def _push_attempt(self, client: httpx.Client) -> None:
        try:
            changed = self.attempt.stat().st_mtime_ns
            body = self.attempt.read_bytes()
        except FileNotFoundError:
            return
        if changed == self.attempt_shown:
            return
        if len(body) > ATTEMPT_BYTES:
            raise ValueError(f"The live attempt frame {self.attempt} has {len(body)} bytes, more than {ATTEMPT_BYTES}")
        client.put("/attempt.jpg", content=body, headers={"Content-Type": "image/jpeg"}).raise_for_status()
        self.attempt_shown = changed


def _put_document(client: httpx.Client, path: str, document: dict[str, Any]) -> None:
    body = gzip.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"))
    client.put(path, content=body).raise_for_status()


def stream_snapshot(settings: UISettings) -> dict[str, Any]:
    return {
        "metadata": metadata(),
        "server": {"version": DATENWISSENSCHAFTEN_VERSION, "release": settings.release, "twitch": settings.twitch},
    }
