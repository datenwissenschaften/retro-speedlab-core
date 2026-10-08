import gzip
import json
from pathlib import Path

import httpx

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

    uploads = [(path, body) for method, path, body in requests if method == "PUT" and "/media/" in path]
    pushed = json.loads(gzip.decompress(next(body for method, path, body in requests if path == "/live/feed")))
    assert [path for path, _ in uploads] == [f"/live/media/{key}.json"]
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
