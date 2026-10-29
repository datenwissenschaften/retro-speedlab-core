import gzip
import json
from pathlib import Path

import httpx

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui import live, relay, telemetry
from datenwissenschaften.ui.live import LiveFeed

RESULT = {"start": "PowerOn", "reached": "Play", "seconds": 1.0, "verified": 1}
SETTINGS = UISettings(True, "2026.10.08-1", True)


def backend(held: list[str], requests: list[tuple[str, str, bytes]]) -> httpx.Client:
    def answer(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path, request.content))
        if request.method == "GET" and request.url.path == "/live/media":
            return httpx.Response(200, json=held)
        return httpx.Response(204)

    return httpx.Client(base_url="https://api.test/live", transport=httpx.MockTransport(answer))


def test_the_relay_uploads_each_attempt_once_before_the_feed_names_it(monkeypatch, tmp_path: Path):
    feed = LiveFeed()
    monkeypatch.setattr(live, "encode_video", lambda jpegs, frame_rate: b"mp4")
    monkeypatch.setattr(relay, "live_feed", feed)
    feed.record(b"jpeg", {"action": "right"})
    feed.finish_episode(1, 60.0, RESULT)
    key = feed.latest_episode()["episode"]["key"]
    requests: list[tuple[str, str, bytes]] = []
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS)

    pusher._push_feed(backend([f"{key}.mp4"], requests))
    pusher._push_feed(backend([], requests))

    uploads = [(path, body) for method, path, body in requests if method == "PUT" and "/parts/" in path]
    pushed = json.loads(gzip.decompress(next(body for method, path, body in requests if path == "/live/feed")))
    assert [path for path, _ in uploads] == [f"/live/media/{key}.json/parts/0"]
    assert json.loads(gzip.decompress(uploads[0][1])) == [{"action": "right"}]
    assert pushed["episode"]["key"] == key
    assert [method for method, path, _ in requests if path == "/live/media"] == ["GET", "PUT"]


def test_large_media_goes_up_in_parts_below_the_waf_body_limit(monkeypatch):
    requests: list[tuple[str, str, bytes]] = []
    monkeypatch.setattr(relay, "PART_BYTES", 4)

    relay._upload(backend([], requests), "name.mp4", b"0123456789")

    assert [(path, body) for _, path, body in requests] == [
        ("/live/media/name.mp4/parts/0", b"0123"),
        ("/live/media/name.mp4/parts/1", b"4567"),
        ("/live/media/name.mp4/parts/2", b"89"),
    ]


def test_the_snapshot_names_the_release_and_carries_the_published_metadata(monkeypatch):
    monkeypatch.setattr(relay, "metadata", lambda: {"speedrun": {"frames": 3}})

    snapshot = relay.stream_snapshot(SETTINGS)

    assert snapshot["server"]["release"] == SETTINGS.release
    assert snapshot["server"]["twitch"] is True
    assert snapshot["metadata"] == {"speedrun": {"frames": 3}}


def test_the_snapshot_goes_up_gzipped(monkeypatch):
    monkeypatch.setattr(relay, "stream_snapshot", lambda settings: {"metadata": {}})
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS)._push_snapshot(backend([], requests))

    assert [(method, path) for method, path, _ in requests] == [("PUT", "/live/snapshot")]
    assert json.loads(gzip.decompress(requests[0][2])) == {"metadata": {}}


def test_the_relay_keeps_pushing_after_the_backend_is_unreachable_until_it_stops(monkeypatch, tmp_path: Path):
    outcomes = [httpx.ConnectError("down"), None, None]
    pushes: list[httpx.Client] = []
    pusher = relay.LiveRelay("https://api.test", "key", SETTINGS)
    monkeypatch.setattr(relay, "RELAY_SECONDS", 0.0)

    def push(client: httpx.Client) -> None:
        pushes.append(client)
        outcome = outcomes.pop(0)
        if not outcomes:
            pusher.stop()
        if outcome is not None:
            raise outcome

    pusher._repeat(push)

    assert len(pushes) == 3


def test_the_relay_starts_one_thread_for_the_snapshot_and_one_for_the_feed(monkeypatch, tmp_path: Path):
    started: list[str] = []

    class Thread:
        def __init__(self, target, args, name: str, daemon: bool) -> None:
            self.name = name

        def start(self) -> None:
            started.append(self.name)

    monkeypatch.setattr(relay.threading, "Thread", Thread)

    relay.LiveRelay("https://api.test", "key", SETTINGS).start()

    assert started == ["relay_push_snapshot", "relay_push_feed"]


def test_media_that_vanished_before_the_upload_is_skipped(monkeypatch, tmp_path: Path):
    feed = LiveFeed()
    monkeypatch.setattr(feed, "media_names", lambda: {"gone.mp4"})
    monkeypatch.setattr(relay, "live_feed", feed)
    requests: list[tuple[str, str, bytes]] = []

    relay.LiveRelay("https://api.test", "key", SETTINGS)._push_feed(backend([], requests))

    assert not [path for _, path, _ in requests if "/parts/" in path]


def test_published_metadata_replaces_its_section():
    telemetry.publish_metadata("speedrun", {"frames": 9})
    telemetry.publish_metadata("speedrun", {"frames": 7})

    assert telemetry.metadata()["speedrun"] == {"frames": 7}
