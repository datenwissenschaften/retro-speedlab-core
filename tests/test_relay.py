import gzip
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from datenwissenschaften.settings import UISettings, UploadSettings
from datenwissenschaften.speedrun.mosaic import tile
from datenwissenschaften.ui import relay, telemetry, video_upload
from datenwissenschaften.ui.video_upload import VideoUpload

RELAY_URL = "https://api.test"
SETTINGS = UISettings(True, True, "2026.10.08-1", True)
BACKGROUND = UISettings(True, False, "2026.10.08-1", True)


headers: list[httpx.Headers] = []


def api(requests: list[tuple[str, str, bytes]], base_url: str) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        headers.append(request.headers)
        return httpx.Response(404 if request.method == "HEAD" else 204)

    return httpx.Client(base_url=base_url, transport=httpx.MockTransport(answer))


def mocked(answer: Callable[[httpx.Request], httpx.Response]) -> Callable[[str, str], httpx.Client]:
    return lambda url, api_key: httpx.Client(base_url=url, transport=httpx.MockTransport(answer))


def test_the_movie_goes_up_in_parts_below_the_waf_body_limit(monkeypatch, tmp_path: Path):
    requests: list[tuple[str, str, bytes]] = []
    movie = tmp_path / "screen.mp4"
    movie.write_bytes(b"0123456789")
    monkeypatch.setattr(video_upload, "PART_BYTES", 4)
    monkeypatch.setattr(video_upload, "relay_client", lambda url, api_key: api(requests, url))

    game = f"https://api.test{relay.game_path('comet_courier')}"
    VideoUpload(game, "key", "movie.mp4").put(movie, "abc", {"movie-ram.json": {"every": 6}})

    assert [(path, body) for _, path, body in requests] == [
        ("/games/comet_courier/movie.mp4", b""),
        ("/games/comet_courier/movie.mp4/parts/0", b"0123"),
        ("/games/comet_courier/movie.mp4/parts/1", b"4567"),
        ("/games/comet_courier/movie.mp4/parts/2", b"89"),
        ("/games/comet_courier/movie-ram.json", b'{"version":"abc","every":6}'),
    ]


def test_a_part_that_times_out_is_sent_again_alone(monkeypatch, tmp_path: Path):
    sent: list[str] = []
    timeouts = [httpx.WriteTimeout("slow uplink")]

    def answer(request: httpx.Request) -> httpx.Response:
        if request.method == "HEAD":
            return httpx.Response(404)
        sent.append(request.url.path)
        if request.url.path.endswith("/parts/1") and timeouts:
            raise timeouts.pop()
        return httpx.Response(204)

    movie = tmp_path / "screen.mp4"
    movie.write_bytes(b"0123456789")
    monkeypatch.setattr(video_upload, "PART_BYTES", 4)
    monkeypatch.setattr(video_upload, "FIRST_WAIT_SECONDS", 0.0)
    monkeypatch.setattr(video_upload, "relay_client", mocked(answer))

    VideoUpload(RELAY_URL, "key", "movie.mp4").put(movie, "abc", {})

    assert sent == ["/movie.mp4/parts/0", "/movie.mp4/parts/1", "/movie.mp4/parts/1", "/movie.mp4/parts/2"]


def test_a_video_the_backend_holds_is_not_sent_again(monkeypatch, tmp_path: Path):
    sent: list[tuple[str, str]] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append((request.method, request.url.path))
        return httpx.Response(200, headers={"ETag": '"abc"'}) if request.method == "HEAD" else httpx.Response(204)

    movie = tmp_path / "screen.mp4"
    movie.write_bytes(b"0123456789")
    monkeypatch.setattr(video_upload, "relay_client", mocked(answer))

    VideoUpload(RELAY_URL, "key", "movie.mp4").put(movie, "abc", {"movie-ram.json": {"every": 6}})

    assert sent == [("HEAD", "/movie.mp4"), ("PUT", "/movie-ram.json")]


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

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path, "walk")._push_snapshot(api(requests, RELAY_URL))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/snapshot")]
    assert json.loads(gzip.decompress(requests[0][2])) == {"metadata": {}}


def test_a_game_off_the_stream_reports_its_run_to_its_own_page_only(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"metadata": {}})
    requests: list[tuple[str, str, bytes]] = []
    pusher = relay.LiveRelay("https://api.test", "key", BACKGROUND, tmp_path, "walk")

    pusher._push_snapshot(api(requests, RELAY_URL))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/games/walk/snapshot")]


def test_the_relay_keeps_pushing_after_the_backend_is_unreachable_until_it_stops(monkeypatch, tmp_path: Path):
    outcomes = [httpx.ConnectError("down"), None, None]
    pushes: list[httpx.Client] = []
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path, "walk")

    def push(client: httpx.Client) -> None:
        pushes.append(client)
        outcome = outcomes.pop(0)
        if not outcomes:
            pusher.stop()
        if outcome is not None:
            raise outcome

    pusher._repeat(push, 0.0)

    assert len(pushes) == 3


def test_the_relay_starts_one_thread_each_for_the_snapshot_and_only_on_stream_the_attempts(monkeypatch, tmp_path: Path):
    started: list[str] = []

    class Thread:
        def __init__(self, target, args, name: str, daemon: bool) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(relay.threading, "Thread", Thread)

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path, "walk").start()
    relay.LiveRelay("https://api.test", "key", BACKGROUND, tmp_path, "walk").start()

    assert started == ["relay_push_snapshot", "relay_push_attempts", "relay_push_snapshot"]


def test_published_metadata_replaces_its_section():
    telemetry.publish_metadata("speedrun", {"frames": 9})
    telemetry.publish_metadata("speedrun", {"frames": 7})

    assert telemetry.metadata()["speedrun"] == {"frames": 7}


def test_the_live_attempts_mosaic_goes_up_once_per_change_with_its_seeds(tmp_path: Path):
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path, "walk")
    requests: list[tuple[str, str, bytes]] = []

    pusher._push_attempts(api(requests, RELAY_URL))
    for seed in (9, 3):
        (tmp_path / f"{seed}.jpg").write_bytes(tile(np.zeros((8, 8, 3), np.uint8)))
    pusher._push_attempts(api(requests, RELAY_URL))
    pusher._push_attempts(api(requests, RELAY_URL))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/attempts.jpg")]
    assert requests[0][2][:2] == b"\xff\xd8" and headers[-1]["X-Seeds"] == "3,9"


def test_an_oversized_attempts_mosaic_fails_fast(tmp_path: Path, monkeypatch):
    (tmp_path / "1.jpg").write_bytes(tile(np.zeros((8, 8, 3), np.uint8)))
    monkeypatch.setattr(relay, "ATTEMPTS_BYTES", 3)

    with pytest.raises(ValueError, match="more than 3"):
        relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path, "walk")._push_attempts(api([], RELAY_URL))
