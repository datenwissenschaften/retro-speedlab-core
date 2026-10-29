import gzip
import json
import threading
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.media import STATUSES_SUFFIX, VIDEO_SUFFIX
from datenwissenschaften.ui.telemetry import metadata

RELAY_SECONDS = 3.0
ATTEMPT_SECONDS = 0.5
ATTEMPT_BYTES = 256 * 1024
PART_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 120.0


def _datenwissenschaften_version() -> str:
    try:
        return version("datenwissenschaften")
    except PackageNotFoundError:
        return "DEVELOPMENT"


DATENWISSENSCHAFTEN_VERSION = _datenwissenschaften_version()


class LiveRelay:
    def __init__(self, url: str, api_key: str, ui: UISettings, attempt: Path) -> None:
        self.url = f"{url}/live"
        self.api_key = api_key
        self.ui = ui
        self.attempt = attempt
        self.attempt_shown: int | None = None
        self.held: set[str] | None = None
        self.kept: set[str] = set()
        self.stopped = threading.Event()

    def start(self) -> None:
        pushes = (
            (self._push_snapshot, RELAY_SECONDS),
            (self._push_feed, RELAY_SECONDS),
            (self._push_attempt, ATTEMPT_SECONDS),
        )
        for push, seconds in pushes:
            name = f"relay{push.__name__}"
            threading.Thread(target=self._repeat, args=(push, seconds), name=name, daemon=True).start()

    def stop(self) -> None:
        self.stopped.set()

    def _repeat(self, push: Callable[[httpx.Client], None], seconds: float) -> None:
        failing = False
        with httpx.Client(base_url=self.url, headers={"X-API-Key": self.api_key}, timeout=TIMEOUT_SECONDS) as client:
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

    def _push_feed(self, client: httpx.Client) -> None:
        if self.held is None:
            self.held = set(client.get("/media").raise_for_status().json())
        wanted = live_feed.media_names()
        for name in sorted(wanted - self.held):
            try:
                body = live_feed.media(name)
            except KeyError:
                continue
            _upload(client, name, gzip.compress(body) if name.endswith(STATUSES_SUFFIX) else body)
            self.held.add(name)
        _put_document(client, "/feed", _held_feed(live_feed.latest_episode(), self.held))
        if wanted != self.kept:
            client.put("/media", json=sorted(wanted)).raise_for_status()
            self.kept = wanted


def _held_feed(feed: dict[str, Any], held: set[str]) -> dict[str, Any]:
    episode = feed["episode"]
    ready = episode is not None and {episode["key"] + VIDEO_SUFFIX, episode["key"] + STATUSES_SUFFIX} <= held
    return {**feed, "episode": episode if ready else None}


def _upload(client: httpx.Client, name: str, body: bytes) -> None:
    starts = range(0, len(body), PART_BYTES)
    for index, start in enumerate(starts):
        last = index == len(starts) - 1
        part = body[start : start + PART_BYTES]
        client.put(f"/media/{name}/parts/{index}", params={"last": last}, content=part).raise_for_status()


def _put_document(client: httpx.Client, path: str, document: dict[str, Any]) -> None:
    body = gzip.compress(json.dumps(document, separators=(",", ":")).encode("utf-8"))
    client.put(path, content=body).raise_for_status()


def stream_snapshot(settings: UISettings) -> dict[str, Any]:
    return {
        "metadata": metadata(),
        "server": {"version": DATENWISSENSCHAFTEN_VERSION, "release": settings.release, "twitch": settings.twitch},
    }
