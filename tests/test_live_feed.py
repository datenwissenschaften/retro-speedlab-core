import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from datenwissenschaften.ui import live
from datenwissenschaften.ui.live import MAX_COMPLETED_EPISODES, LiveFeed
from datenwissenschaften.ui.replay_video import encode_video

RESULT = {"start": "PowerOn", "reached": "Play", "seconds": 1.0, "verified": 1}


@pytest.fixture(autouse=True)
def fake_video(monkeypatch):
    monkeypatch.setattr(live, "encode_video", lambda jpegs, frame_rate: b"mp4:" + b"".join(jpegs))


def play(feed: LiveFeed, episode_id: int, frames: int, result: dict) -> None:
    for index in range(frames):
        feed.record(b"jpeg", {"timesteps": index, "attempt": episode_id, "level": "Level1"})
    feed.finish_episode(episode_id, 60.0, result)


def test_the_latest_attempts_are_kept_as_a_video_and_their_decisions_under_one_content_key():
    feed = LiveFeed()
    generation = feed.latest_episode()["generation"]
    assert feed.latest_episode() == {"generation": generation, "episode": None}
    for episode_id in range(1, MAX_COMPLETED_EPISODES + 3):
        play(feed, episode_id, episode_id, RESULT)

    newest = feed.latest_episode()["episode"]
    key = newest["key"]

    assert feed.media(f"{key}.mp4") == b"mp4:" + b"jpeg" * newest["id"]
    assert [status["timesteps"] for status in json.loads(feed.media(f"{key}.json"))] == list(range(newest["id"]))
    assert len(feed.media_names()) == 2 * MAX_COMPLETED_EPISODES
    with pytest.raises(KeyError):
        feed.media("0" * 32 + ".mp4")


def test_an_episode_without_frames_is_not_kept():
    feed = LiveFeed()
    feed.finish_episode(1, 60.0, RESULT)

    assert feed.latest_episode()["episode"] is None
    assert feed.media_names() == set()


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
