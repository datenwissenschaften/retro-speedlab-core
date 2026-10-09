import gzip
import json
import threading
import time
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.attempts import recorded_attempts
from datenwissenschaften.ui.control import control_metadata, request_model_reset
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.media import STATUSES_SUFFIX, VIDEO_SUFFIX
from datenwissenschaften.ui.persona import persona_tag
from datenwissenschaften.ui.telemetry import get_store

RELAY_SECONDS = 3.0
PART_BYTES = 8 * 1024 * 1024
TIMEOUT_SECONDS = 120.0


def _datenwissenschaften_version() -> str:
    try:
        return version("datenwissenschaften")
    except PackageNotFoundError:
        return "DEVELOPMENT"


DATENWISSENSCHAFTEN_VERSION = _datenwissenschaften_version()


class LiveRelay:
    def __init__(self, url: str, api_key: str, ui: UISettings, record_root: Path) -> None:
        self.url = f"{url}/live"
        self.api_key = api_key
        self.ui = ui
        self.record_root = record_root
        self.held: set[str] | None = None
        self.kept: set[str] = set()

    def start(self) -> None:
        for push in (self._push_snapshot, self._push_feed):
            threading.Thread(target=self._repeat, args=(push,), name=f"relay{push.__name__}", daemon=True).start()

    def _repeat(self, push: Callable[[httpx.Client], None]) -> None:
        failing = False
        with httpx.Client(base_url=self.url, headers={"X-API-Key": self.api_key}, timeout=TIMEOUT_SECONDS) as client:
            while True:
                try:
                    push(client)
                    if failing:
                        logger.info("The live relay reaches the backend again")
                    failing = False
                except httpx.HTTPError as error:
                    if not failing:
                        logger.warning(f"The live relay cannot reach the backend: {error}")
                    failing = True
                time.sleep(RELAY_SECONDS)

    def _push_snapshot(self, client: httpx.Client) -> None:
        _put_document(client, "/snapshot", stream_snapshot(self.ui))
        game = client.get("/reset").raise_for_status().json()["game"]
        if game is None:
            return
        client.delete("/reset").raise_for_status()
        try:
            request_model_reset(game)
        except (RuntimeError, ValueError) as error:
            logger.error(f"The model reset for {game} was refused: {error}")
            return
        logger.warning(f"Model reset requested through the backend for {game}")

    def _push_feed(self, client: httpx.Client) -> None:
        if self.held is None:
            self.held = set(client.get("/media").raise_for_status().json())
        attempts = recorded_attempts(self.record_root)
        wanted = live_feed.media_names() | set(attempts)
        for name in sorted(wanted - self.held):
            try:
                body = attempts[name][1].read_bytes() if name in attempts else live_feed.media(name)
            except (KeyError, FileNotFoundError):
                continue
            if name.endswith(STATUSES_SUFFIX):
                body = gzip.compress(body)
            _upload(client, name, body)
            self.held.add(name)
        feed = _held_feed(live_feed.latest_episode(), self.held)
        _put_document(client, "/feed", {**feed, "attempts": [metadata for metadata, _ in attempts.values()]})
        if wanted != self.kept:
            client.put("/media", json=sorted(wanted)).raise_for_status()
            self.kept = wanted


def _held_feed(feed: dict[str, Any], held: set[str]) -> dict[str, Any]:
    def ready(episode: dict[str, Any] | None) -> bool:
        return episode is not None and {episode["key"] + VIDEO_SUFFIX, episode["key"] + STATUSES_SUFFIX} <= held

    return {
        **feed,
        "episode": feed["episode"] if ready(feed["episode"]) else None,
        "replays": [replay for replay in feed["replays"] if ready(replay)],
    }


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
        **get_store().snapshot(),
        "control": control_metadata(),
        "server": {
            "version": DATENWISSENSCHAFTEN_VERSION,
            "release": settings.release,
            "persona": settings.persona,
            "persona_tag": persona_tag(settings.release),
            "twitch": settings.twitch,
        },
    }
