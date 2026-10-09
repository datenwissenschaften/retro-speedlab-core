import gzip
import json
from pathlib import Path

import httpx
import pytest

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui import live, relay
from datenwissenschaften.ui.live import LiveFeed

RESULT = {"score": 1.0, "succeeded": False, "curriculum": "Play"}
SETTINGS = UISettings(True, 10, "2026.10.08-1", "Retra", True, ("model",))


def backend(held: list[str], requests: list[tuple[str, str, bytes]], reset: str | None) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        if request.method == "GET" and request.url.path == "/live/media":
            return httpx.Response(200, json=held)
        if request.method == "GET" and request.url.path == "/live/reset":
            return httpx.Response(200, json={"game": reset})
        return httpx.Response(204)

    return httpx.Client(base_url="https://api.test/live", transport=httpx.MockTransport(answer))


def test_the_relay_uploads_each_attempt_once_before_the_feed_names_it(monkeypatch, tmp_path: Path):
    feed = LiveFeed()
    monkeypatch.setattr(live, "encode_video", lambda jpegs, frame_rate: b"mp4")
    monkeypatch.setattr(relay, "live_feed", feed)
    feed.record(b"jpeg", {"action": "right"})
    feed.finish_episode(1, 60.0, RESULT, {})
    key = feed.latest_episode()["episode"]["key"]
    requests: list[tuple[str, str, bytes]] = []
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)

    pusher._push_feed(backend([f"{key}.mp4"], requests, None))
    pusher._push_feed(backend([], requests, None))

    uploads = [(path, body) for method, path, body in requests if method == "PUT" and "/parts/" in path]
    pushed = json.loads(gzip.decompress(next(body for method, path, body in requests if path == "/live/feed")))
    assert [path for path, _ in uploads] == [f"/live/media/{key}.json/parts/0"]
    assert json.loads(gzip.decompress(uploads[0][1])) == [{"action": "right"}]
    assert pushed["episode"]["key"] == key
    assert [method for method, path, _ in requests if path == "/live/media"] == ["GET", "PUT"]


def test_a_reset_requested_through_the_backend_reaches_the_training(monkeypatch, tmp_path: Path):
    resets: list[str] = []
    monkeypatch.setattr(relay, "request_model_reset", resets.append)
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"status": "running"})
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_snapshot(backend([], requests, "Game-v0"))

    assert resets == ["Game-v0"]
    assert [(method, path) for method, path, _ in requests] == [
        ("PUT", "/live/snapshot"),
        ("GET", "/live/reset"),
        ("DELETE", "/live/reset"),
    ]


def test_large_media_goes_up_in_parts_below_the_waf_body_limit(monkeypatch):
    requests: list[tuple[str, str, bytes]] = []
    monkeypatch.setattr(relay, "PART_BYTES", 4)

    relay._upload(backend([], requests, None), "name.mp4", b"0123456789")

    assert [(path, body) for _, path, body in requests] == [
        ("/live/media/name.mp4/parts/0", b"0123"),
        ("/live/media/name.mp4/parts/1", b"4567"),
        ("/live/media/name.mp4/parts/2", b"89"),
    ]


def test_recorded_attempts_go_up_with_their_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(relay, "live_feed", LiveFeed())
    (tmp_path / "run.mp4").write_bytes(b"mp4")
    (tmp_path / "run.rollout.json").write_text(json.dumps({"score": 2.0}), encoding="utf-8")
    (tmp_path / "lost.rollout.json").write_text("{}", encoding="utf-8")
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_feed(backend([], requests, None))

    pushed = json.loads(gzip.decompress(next(body for method, path, body in requests if path == "/live/feed")))
    uploads = [body for method, path, body in requests if "/parts/" in path]
    assert [attempt["score"] for attempt in pushed["attempts"]] == [2.0]
    assert uploads == [b"mp4"]


def test_a_refused_reset_leaves_the_training_running(monkeypatch, tmp_path: Path):
    def refuse(game: str) -> None:
        raise ValueError(game)

    monkeypatch.setattr(relay, "request_model_reset", refuse)
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"status": "running"})
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_snapshot(backend([], requests, "Other-v0"))

    assert ("DELETE", "/live/reset") in [(method, path) for method, path, _ in requests]


def test_the_snapshot_names_the_release_and_its_persona():
    server = relay.stream_snapshot(SETTINGS)["server"]

    assert server["release"] == SETTINGS.release
    assert server["persona"] == "Retra"
    assert len(server["persona_tag"]) == 6


class Stop(Exception):
    pass


def test_the_relay_keeps_pushing_after_the_backend_is_unreachable(monkeypatch, tmp_path: Path):
    outcomes = [httpx.ConnectError("down"), None, None]
    pushes: list[httpx.Client] = []

    def push(client: httpx.Client) -> None:
        pushes.append(client)
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome

    def sleep(seconds: float) -> None:
        if not outcomes:
            raise Stop

    monkeypatch.setattr(relay.time, "sleep", sleep)

    with pytest.raises(Stop):
        relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._repeat(push)

    assert len(pushes) == 3


def test_the_relay_starts_one_thread_for_the_snapshot_and_one_for_the_feed(monkeypatch, tmp_path: Path):
    started: list[str] = []

    class Thread:
        def __init__(self, target, args, name: str, daemon: bool) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(relay.threading, "Thread", Thread)

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path).start()

    assert started == ["relay_push_snapshot", "relay_push_feed"]


def test_without_a_pending_reset_only_the_snapshot_goes_up(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"status": "running"})
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_snapshot(backend([], requests, None))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/snapshot"), ("GET", "/live/reset")]


def test_media_that_vanished_before_the_upload_is_skipped(monkeypatch, tmp_path: Path):
    feed = LiveFeed()
    monkeypatch.setattr(feed, "media_names", lambda: {"gone.mp4"})
    monkeypatch.setattr(relay, "live_feed", feed)
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS, tmp_path)._push_feed(backend([], requests, None))

    assert not [path for _, path, _ in requests if "/parts/" in path]
