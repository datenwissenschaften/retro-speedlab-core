import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from datenwissenschaften.route.position import Position
from datenwissenschaften.tas import footage
from datenwissenschaften.tas.footage import compare_cut, movie_cut, movie_field, render_compare, render_movie
from datenwissenschaften.tas.movie import Segment, memory_field, memory_sample
from datenwissenschaften.tas.routes import Leg
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.screen import Cut, Screen
from datenwissenschaften.tas.search import Crossing

COURSE = SimpleNamespace(first="PowerOn", frame_rate=60.0)
LEG = Leg("Level 1", Start("PowerOn", b""), None)
SEGMENTS = [Segment("Play", [[0, 1]] * 2, b"", False), Segment("Boss", [[1, 0]] * 3, b"", False)]
MEMORY = {"every": 6, "keys": ["lives"], "samples": [[3]]}


def render(recordings: tuple[bytes, ...]) -> Path:
    video = Path(render.directory) / "video.mp4"
    video.write_bytes(b"mp4:" + b"".join(recordings))
    return video


def flaky_upload(outcomes: list[bool], uploads: list[tuple]) -> SimpleNamespace:
    def put(video: Path, version: str, attachments: dict) -> None:
        if not outcomes.pop(0):
            raise httpx.ConnectError("down")
        uploads.append((video.read_bytes(), version, attachments))

    return SimpleNamespace(put=put)


def cut(recording: bytes, field: dict) -> Cut:
    return Cut(recording, (recording,), field, {})


def test_the_movie_names_where_each_segment_starts_plays_and_what_it_reaches():
    assert movie_field("PowerOn", 60.0, SEGMENTS) == {
        "frames": 5,
        "seconds": 5 / 60.0,
        "frame_rate": 60.0,
        "segments": [
            {"state": "PowerOn", "reached": "Play", "start_frame": 0, "frames": 2},
            {"state": "Play", "reached": "Boss", "start_frame": 2, "frames": 3},
        ],
    }


def test_a_changed_movie_is_rendered_uploaded_with_its_memory_and_shown_once(tmp_path: Path):
    render.directory = tmp_path
    recording, shown, uploads = tmp_path / "tas.bk2", [], []
    recording.write_bytes(b"bk2")
    movies = Screen("movie", render, flaky_upload([True], uploads), shown.append)
    movies.start()

    movies.queue(movie_cut(COURSE, LEG, SEGMENTS, recording, MEMORY))
    movies.queue(movie_cut(COURSE, LEG, SEGMENTS, recording, MEMORY))
    movies.settle()

    version = hashlib.sha256(b"mp4:bk2").hexdigest()[:12]
    assert uploads == [(b"mp4:bk2", version, {"movie-ram.json": MEMORY})]
    assert shown == [{"version": version, **movie_field("PowerOn", 60.0, SEGMENTS), "part": "Level 1"}]


def test_an_upload_is_retried_until_the_backend_takes_it(tmp_path: Path, monkeypatch):
    render.directory = tmp_path
    monkeypatch.setattr("datenwissenschaften.tas.screen.RETRY_SECONDS", 0.0)
    shown, uploads = [], []
    movies = Screen("movie", render, flaky_upload([False, True], uploads), shown.append)

    movies._present(cut(b"bk2", {"frames": 1}))

    assert len(uploads) == 1 and len(shown) == 1


def test_a_render_overtaken_by_a_newer_one_is_not_uploaded(tmp_path: Path):
    render.directory = tmp_path
    shown: list[dict] = []
    movies = Screen("movie", render, None, shown.append)
    movies.pending = cut(b"newer", {"frames": 2})

    movies._present(cut(b"older", {"frames": 1}))

    assert shown == []


def test_a_video_without_an_upload_is_shown_locally(tmp_path: Path):
    render.directory = tmp_path
    shown: list[dict] = []

    Screen("compare", render, None, shown.append)._present(cut(b"bk2", {"frames": 1}))

    assert shown[0]["frames"] == 1


def test_a_video_that_does_not_render_is_skipped():
    def fail(recordings: tuple[bytes, ...]) -> Path:
        raise subprocess.CalledProcessError(1, ["python"], "", "ffmpeg failed")

    shown: list[dict] = []

    Screen("movie", fail, None, shown.append)._present(cut(b"bk2", {"frames": 1}))

    assert shown == []


def test_the_movie_and_the_comparison_render_from_their_recordings(tmp_path: Path, monkeypatch):
    rendered: list[list[str]] = []

    def render_video(paths: object, recordings: list[Path]) -> list[Path]:
        rendered.append([recording.read_bytes().decode() for recording in recordings])
        return [recording.with_suffix(".mp4") for recording in recordings]

    monkeypatch.setattr(footage, "render_video", render_video)
    monkeypatch.setattr(footage, "stack_videos", lambda left, right, stacked: (left.name, right.name, stacked.name))

    movie = render_movie(SimpleNamespace(), tmp_path, (b"run",))
    compare = render_compare(SimpleNamespace(), tmp_path, (b"old", b"new"))

    assert movie == tmp_path / "screen.mp4"
    assert compare == ("compare-before.mp4", "compare-after.mp4", "compare.mp4")
    assert rendered == [["run"], ["old", "new"]]


def test_a_faster_piece_is_recorded_beside_the_old_one_from_the_same_start(tmp_path: Path, monkeypatch):
    pieces: list[tuple[bytes, int]] = []

    def record_piece(runner: object, start: object, pressed: np.ndarray, movie: Path) -> bytes:
        pieces.append((start, len(pressed)))
        return f"piece{len(pressed)}".encode()

    monkeypatch.setattr(footage, "record_piece", record_piece)
    after = Crossing(SimpleNamespace(state="Boss"), np.zeros((2, 2), np.int8), False, ("Boss",))

    compared = compare_cut(SimpleNamespace(), tmp_path, "start", SEGMENTS[1], after)

    assert pieces == [("start", 3), ("start", 2)]
    assert compared.recordings == (b"piece3", b"piece2")
    assert {key: compared.field[key] for key in ("state", "before_frames", "after_frames")} == {
        "state": "Boss",
        "before_frames": 3,
        "after_frames": 2,
    }
    assert "at" in compared.field


def test_the_movie_memory_names_each_value_once_and_fails_when_the_names_change():
    located = SimpleNamespace(describe=lambda: {"lives": 3}, position=lambda: Position(2, 10, 20, 5, False))
    lost = SimpleNamespace(describe=lambda: {"lives": 2}, position=lambda: None)

    memory = memory_field([memory_sample(located), memory_sample(lost)])

    assert memory == {
        "every": 6,
        "keys": ["lives", "area", "x", "y", "height"],
        "samples": [[3, 2, 10, 20, 5], [2, None, None, None, None]],
    }
    with pytest.raises(ValueError, match="memory"):
        memory_field([{"lives": 3}, {"score": 1}])
