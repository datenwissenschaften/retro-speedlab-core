import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from datenwissenschaften.reference import fetch, paths, speedrun_com
from datenwissenschaften.settings import SpeedrunComSource

SOURCE = SpeedrunComSource("snake", "warpless", {"players": "one"})
RUN_URL = "https://www.youtube.com/watch?v=run"


def leaderboard(monkeypatch: pytest.MonkeyPatch, runs: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    asked: list[tuple[str, dict[str, Any]]] = []

    def answer(url: str, params: dict[str, Any], timeout: int) -> SimpleNamespace:
        asked.append((url, params))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"data": {"runs": runs}})

    monkeypatch.setattr(speedrun_com.httpx, "get", answer)
    return asked


def place(videos: dict[str, Any] | None) -> dict[str, Any]:
    return {"place": 1, "run": {"videos": videos}}


def test_the_reference_candidates_are_the_videos_of_the_top_runs_in_their_order(monkeypatch: pytest.MonkeyPatch):
    second = "https://www.twitch.tv/videos/2"
    asked = leaderboard(
        monkeypatch, [place(None), place({"text": "on request"}), place({"links": [{"uri": RUN_URL}, {"uri": second}]})]
    )

    assert speedrun_com.video_urls(SOURCE) == [RUN_URL, second]
    assert asked == [
        (
            "https://www.speedrun.com/api/v1/leaderboards/snake/category/warpless",
            {"top": speedrun_com.TOP_PLACES, "var-players": "one"},
        )
    ]


def test_a_video_that_does_not_download_makes_way_for_the_next_and_none_leaves_no_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def download(url: str, directory: Path) -> None:
        if url == RUN_URL:
            raise subprocess.CalledProcessError(1, ["yt_dlp"])

    monkeypatch.setattr(fetch, "download", download)
    assert fetch.downloaded([RUN_URL, "https://www.twitch.tv/videos/2"], tmp_path) == "https://www.twitch.tv/videos/2"
    assert fetch.downloaded([RUN_URL], tmp_path) is None


def test_the_video_is_downloaded_with_yt_dlp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", lambda command, check: commands.append(command))

    speedrun_com.download(RUN_URL, tmp_path / "reference")

    assert (tmp_path / "reference").is_dir()
    assert commands == [
        [sys.executable, "-m", "yt_dlp", "--quiet", "--no-playlist", "-f", speedrun_com.VIDEO_FORMAT]
        + ["-o", str(tmp_path / "reference" / "video.%(ext)s"), RUN_URL]
    ]


def test_a_reference_with_frames_is_not_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(paths=SimpleNamespace(cache_dir=tmp_path))
    (tmp_path / paths.REFERENCE_DIR / paths.FRAMES_DIR).mkdir(parents=True)
    monkeypatch.setattr(fetch, "video_urls", lambda source: pytest.fail("fetched again"))

    fetch.ensure_reference(settings)


def write_run(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    video = cv2.VideoWriter(str(directory / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 32))
    for _ in range(8):
        video.write(np.zeros((32, 32, 3), np.uint8))
    video.release()


def test_a_missing_reference_is_downloaded_and_split_into_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(fetch, "video_urls", lambda source: [RUN_URL] if source == SOURCE else pytest.fail("other run"))
    monkeypatch.setattr(fetch, "download", lambda url, directory: write_run(directory))
    settings = SimpleNamespace(
        paths=SimpleNamespace(cache_dir=tmp_path), speedrun=SimpleNamespace(game=SimpleNamespace(speedrun_com=SOURCE))
    )

    fetch.ensure_reference(settings)

    reference = tmp_path / paths.REFERENCE_DIR
    assert paths.video_path(settings) == reference / "video.avi"
    assert sorted(frame.name for frame in (reference / paths.FRAMES_DIR).glob("*.png"))[:2] == [
        "0000.00s.png",
        "0000.25s.png",
    ]


def test_a_reference_without_one_video_fails(tmp_path: Path):
    settings = SimpleNamespace(paths=SimpleNamespace(cache_dir=tmp_path))

    with pytest.raises(FileNotFoundError, match="not one"):
        paths.video_path(settings)
