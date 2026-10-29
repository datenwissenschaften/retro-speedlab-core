import gzip
import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from datenwissenschaften.settings import UISettings, UploadSettings
from datenwissenschaften.tas.mosaic import tile
from datenwissenschaften.ui import relay, telemetry, video_upload
from datenwissenschaften.ui.video_upload import VideoUpload

SETTINGS = UISettings(True, "2026.10.08-1", True)


headers: list[httpx.Headers] = []


def backend(requests: list[tuple[str, str, bytes]]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        headers.append(request.headers)
        return httpx.Response(204)

    return httpx.Client(base_url="https://api.test/live", transport=httpx.MockTransport(answer))


def test_the_movie_goes_up_in_parts_below_the_waf_body_limit(monkeypatch, tmp_path: Path):
    requests: list[tuple[str, str, bytes]] = []
    movie = tmp_path / "screen.mp4"
    movie.write_bytes(b"0123456789")
    monkeypatch.setattr(video_upload, "PART_BYTES", 4)
    monkeypatch.setattr(video_upload, "backend_client", lambda url, api_key: backend(requests))

    VideoUpload("https://api.test", "key", "movie.mp4").put(movie, "abc", {"movie-ram.json": {"every": 6}})

    assert [(path, body) for _, path, body in requests] == [
        ("/live/movie.mp4/parts/0", b"0123"),
        ("/live/movie.mp4/parts/1", b"4567"),
        ("/live/movie.mp4/parts/2", b"89"),
        ("/live/movie-ram.json", b'{"version":"abc","every":6}'),
    ]


def test_an_oversized_movie_fails_fast(monkeypatch, tmp_path: Path):
    movie = tmp_path / "screen.mp4"
    movie.write_bytes(b"0123")
    monkeypatch.setattr(video_upload, "VIDEO_BYTES", 3)

    with pytest.raises(ValueError, match="more than 3"):
        VideoUpload("https://api.test", "key", "compare.mp4").put(movie, "abc", {})


def test_the_relay_key_is_required():
    with pytest.raises(RuntimeError, match="upload.api_key"):
        relay.relay_key(UploadSettings("https://api.test", None))

    assert relay.relay_key(UploadSettings("https://api.test", "key")) == "key"


def test_the_snapshot_names_the_release_and_carries_the_published_metadata(monkeypatch):
    monkeypatch.setattr(relay, "metadata", lambda: {"speedrun": {"frames": 3}})

    snapshot = relay.stream_snapshot(SETTINGS)

    assert snapshot["server"]["release"] == SETTINGS.release
    assert snapshot["server"]["twitch"] is True
    assert snapshot["metadata"] == {"speedrun": {"frames": 3}}


def test_the_snapshot_goes_up_gzipped(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"metadata": {}})
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_snapshot(backend(requests))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/snapshot")]
    assert json.loads(gzip.decompress(requests[0][2])) == {"metadata": {}}


def test_the_relay_keeps_pushing_after_the_backend_is_unreachable_until_it_stops(monkeypatch, tmp_path: Path):
    outcomes = [httpx.ConnectError("down"), None, None]
    pushes: list[httpx.Client] = []
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)

    def push(client: httpx.Client) -> None:
        pushes.append(client)
        outcome = outcomes.pop(0)
        if not outcomes:
            pusher.stop()
        if outcome is not None:
            raise outcome

    pusher._repeat(push, 0.0)

    assert len(pushes) == 3


def test_the_relay_starts_one_thread_each_for_the_snapshot_and_the_attempts(monkeypatch, tmp_path: Path):
    started: list[str] = []

    class Thread:
        def __init__(self, target, args, name: str, daemon: bool) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(relay.threading, "Thread", Thread)

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path).start()

    assert started == ["relay_push_snapshot", "relay_push_attempts"]


def test_published_metadata_replaces_its_section():
    telemetry.publish_metadata("speedrun", {"frames": 9})
    telemetry.publish_metadata("speedrun", {"frames": 7})

    assert telemetry.metadata()["speedrun"] == {"frames": 7}


def test_the_live_attempts_mosaic_goes_up_once_per_change_with_its_seeds(tmp_path: Path):
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)
    requests: list[tuple[str, str, bytes]] = []

    pusher._push_attempts(backend(requests))
    for seed in (9, 3):
        (tmp_path / f"{seed}.jpg").write_bytes(tile(np.zeros((8, 8, 3), np.uint8)))
    pusher._push_attempts(backend(requests))
    pusher._push_attempts(backend(requests))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/attempts.jpg")]
    assert requests[0][2][:2] == b"\xff\xd8" and headers[-1]["X-Seeds"] == "3,9"


def test_an_oversized_attempts_mosaic_fails_fast(tmp_path: Path, monkeypatch):
    (tmp_path / "1.jpg").write_bytes(tile(np.zeros((8, 8, 3), np.uint8)))
    monkeypatch.setattr(relay, "ATTEMPTS_BYTES", 3)

    with pytest.raises(ValueError, match="more than 3"):
        relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_attempts(backend([]))
