import subprocess
from pathlib import Path

import cv2
import msgspec
import numpy as np
import pytest

from datenwissenschaften.ui import live
from datenwissenschaften.ui.live import MAX_COMPLETED_EPISODES, MAX_STATUSES_PER_REQUEST, LiveFeed
from datenwissenschaften.ui.replay_video import encode_video

LOST = {"score": 1.0, "won": False, "new_best": False, "full_run": False, "succeeded": False, "curriculum": "Play"}
SUCCEEDED = {**LOST, "succeeded": True}


@pytest.fixture(autouse=True)
def fake_video(monkeypatch):
    monkeypatch.setattr(live, "encode_video", lambda jpegs, frame_rate: b"mp4:" + b"".join(jpegs))


def play(feed: LiveFeed, episode_id: int, frames: int, result: dict) -> None:
    for index in range(frames):
        feed.record(b"jpeg", {"timesteps": index, "attempt": episode_id, "level": "Level1"})
    feed.finish_episode(episode_id, 60.0, result, {})


def test_the_latest_attempts_are_kept_as_a_video_with_their_decisions_in_chunks():
    feed = LiveFeed()
    generation = feed.latest_episode()["generation"]
    empty = {"generation": generation, "episode": None, "replays": [], "summary": {}}
    assert feed.latest_episode() == empty
    for episode_id in range(1, MAX_COMPLETED_EPISODES + 3):
        play(feed, episode_id, MAX_STATUSES_PER_REQUEST + 5, LOST)

    newest = MAX_COMPLETED_EPISODES + 2
    rest = feed.episode_statuses(generation, newest, MAX_STATUSES_PER_REQUEST)

    assert len(feed.episode_statuses(generation, newest, 0)) == MAX_STATUSES_PER_REQUEST
    first_of_rest = MAX_STATUSES_PER_REQUEST
    assert [status["timesteps"] for status in rest] == list(range(first_of_rest, first_of_rest + 5))
    assert feed.episode_video(generation, newest).startswith(b"mp4:jpeg")
    assert feed.latest_episode()["episode"]["frame_count"] == MAX_STATUSES_PER_REQUEST + 5
    with pytest.raises(KeyError):
        feed.episode_video(generation, 2)


def test_clearing_the_live_feed_starts_a_new_generation_without_old_attempts():
    feed = LiveFeed()
    play(feed, 1, 1, LOST)
    old = feed.latest_episode()["generation"]

    feed.clear()
    play(feed, 1, 1, {**LOST, "score": 3.0})

    assert feed.latest_episode()["generation"] != old
    with pytest.raises(KeyError):
        feed.episode_video(old, 1)


CURRICULA = frozenset({"Play", "Grow", "Heavy"})


def test_the_shortest_success_of_every_curriculum_state_stays_available_as_a_replay():
    feed = LiveFeed()
    attempts = [("Play", 5), ("Grow", 4), ("Play", 2), ("Grow", 3), ("Heavy", 6), ("Play", 3)]
    for episode_id, (curriculum, frames) in enumerate(attempts, start=1):
        play(feed, episode_id, frames, {**SUCCEEDED, "curriculum": curriculum})

    replays = [(replay["result"]["curriculum"], replay["id"]) for replay in feed.latest_episode()["replays"]]
    assert replays == [("Play", 3), ("Grow", 4), ("Heavy", 5)]


def test_the_best_replays_survive_a_restart_on_disk_and_go_with_a_reset(tmp_path: Path):
    feed = LiveFeed()
    feed.keep_replays_in(tmp_path / "replays", CURRICULA)
    for episode_id, (curriculum, frames) in enumerate([("Play", 3), ("Grow", 2), ("Play", 2)], start=1):
        play(feed, episode_id, frames, {**SUCCEEDED, "curriculum": curriculum})

    restarted = LiveFeed()
    restarted.keep_replays_in(tmp_path / "replays", CURRICULA)
    generation = restarted.latest_episode()["generation"]

    replays = sorted((replay["result"]["curriculum"], replay["id"]) for replay in restarted.latest_episode()["replays"])
    assert replays == [("Grow", 2), ("Play", 3)]
    assert restarted.episode_video(generation, 3) == b"mp4:jpegjpeg"
    restarted.clear()
    assert list((tmp_path / "replays").iterdir()) == []


def test_replays_stored_as_frames_are_deleted_instead_of_read(tmp_path: Path):
    old = tmp_path / "replays" / "Play.replay"
    old.parent.mkdir()
    old.write_bytes(msgspec.msgpack.encode({"id": 1, "frame_rate": 60.0, "result": LOST, "frames": []}))

    feed = LiveFeed()
    feed.keep_replays_in(old.parent, CURRICULA)

    assert feed.latest_episode()["replays"] == []
    assert not old.exists()


def test_frames_are_encoded_into_a_playable_h264_video(tmp_path: Path):
    frames = [cv2.imencode(".jpg", np.full((224, 240, 3), shade, np.uint8))[1].tobytes() for shade in range(0, 240, 4)]

    video = tmp_path / "replay.mp4"
    video.write_bytes(encode_video(frames, 60.0))
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,nb_read_frames,width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == f"h264,240,224,{len(frames)}"


def test_failed_attempts_never_become_replays_and_the_shortest_success_wins():
    feed = LiveFeed()
    play(feed, 1, 1, {**LOST, "score": 50.0, "curriculum": "Door"})
    play(feed, 2, 4, {**SUCCEEDED, "score": 1.0, "curriculum": "Heavy"})
    play(feed, 3, 2, {**SUCCEEDED, "score": 0.5, "curriculum": "Heavy"})
    play(feed, 4, 3, {**SUCCEEDED, "score": 9.0, "curriculum": "Heavy"})

    assert [replay["id"] for replay in feed.latest_episode()["replays"]] == [3]


def test_stored_failed_replays_are_deleted_at_start(tmp_path: Path):
    old = tmp_path / "replays" / "Heavy.replay"
    old.parent.mkdir()
    old.write_bytes(msgspec.msgpack.encode({"id": 1, "frame_rate": 60.0, "result": LOST, "statuses": [], "video": b""}))

    feed = LiveFeed()
    feed.keep_replays_in(old.parent, CURRICULA)

    assert feed.latest_episode()["replays"] == []
    assert not old.exists()


def test_replays_of_states_the_curriculum_no_longer_has_are_deleted(tmp_path: Path):
    feed = LiveFeed()
    feed.keep_replays_in(tmp_path / "replays", CURRICULA)
    play(feed, 1, 1, {**SUCCEEDED, "curriculum": "Grow"})
    play(feed, 2, 1, {**SUCCEEDED, "curriculum": "Heavy"})

    feed.keep_replays_in(tmp_path / "replays", frozenset({"Heavy"}))

    assert [replay["result"]["curriculum"] for replay in feed.latest_episode()["replays"]] == ["Heavy"]
    assert [path.name for path in (tmp_path / "replays").iterdir()] == ["Heavy.replay"]


def test_the_replays_of_a_beaten_level_s_states_are_dropped(tmp_path: Path):
    feed = LiveFeed()
    feed.keep_replays_in(tmp_path / "replays", CURRICULA)
    for episode_id, curriculum in enumerate(("Play", "Grow", "Heavy"), start=1):
        play(feed, episode_id, 1, {**SUCCEEDED, "curriculum": curriculum})

    feed.drop_replays(("Play", "Grow"))

    assert [replay["result"]["curriculum"] for replay in feed.latest_episode()["replays"]] == ["Heavy"]
    assert [path.name for path in (tmp_path / "replays").iterdir()] == ["Heavy.replay"]
