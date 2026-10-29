import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx

from datenwissenschaften.tas import screen
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.screen import Cut, MovieScreen, movie_field

COURSE = SimpleNamespace(first="PowerOn", frame_rate=60.0)
SEGMENTS = [Segment("Play", [[0, 1]] * 2, b"", False), Segment("Boss", [[1, 0]] * 3, b"", False)]


def render(paths: object, recording: Path) -> Path:
    video = recording.with_suffix(".mp4")
    video.write_bytes(b"mp4:" + recording.read_bytes())
    return video


def flaky_upload(outcomes: list[bool], uploads: list[bytes]) -> SimpleNamespace:
    def put(movie: Path) -> None:
        if not outcomes.pop(0):
            raise httpx.ConnectError("down")
        uploads.append(movie.read_bytes())

    return SimpleNamespace(put=put)


def test_the_movie_names_the_state_each_segment_plays_in_and_where_it_starts():
    assert movie_field("PowerOn", 60.0, SEGMENTS) == {
        "frames": 5,
        "seconds": 5 / 60.0,
        "frame_rate": 60.0,
        "segments": [
            {"state": "PowerOn", "start_frame": 0, "frames": 2},
            {"state": "Play", "start_frame": 2, "frames": 3},
        ],
    }


def test_a_changed_movie_is_rendered_uploaded_and_shown_once(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(screen, "render_video", render)
    recording, shown, uploads = tmp_path / "tas.bk2", [], []
    recording.write_bytes(b"bk2")
    movies = MovieScreen(SimpleNamespace(), tmp_path, COURSE, flaky_upload([True], uploads), shown.append)
    movies.start()

    movies.screen([], recording)
    movies.screen(SEGMENTS, recording)
    movies.screen(SEGMENTS, recording)
    video = movies.settle()

    assert video == tmp_path / "screen.mp4"
    assert uploads == [b"mp4:bk2"]
    assert shown == [{"version": hashlib.sha256(b"mp4:bk2").hexdigest()[:12], **movie_field("PowerOn", 60.0, SEGMENTS)}]


def test_an_upload_is_retried_until_the_backend_takes_it(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(screen, "render_video", render)
    monkeypatch.setattr(screen, "RETRY_SECONDS", 0.0)
    shown, uploads = [], []
    movies = MovieScreen(SimpleNamespace(), tmp_path, COURSE, flaky_upload([False, True], uploads), shown.append)

    movies._present(Cut(b"bk2", {"frames": 1}))

    assert uploads == [b"mp4:bk2"] and len(shown) == 1


def test_a_render_overtaken_by_a_newer_movie_is_not_uploaded(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(screen, "render_video", render)
    shown: list[dict] = []
    movies = MovieScreen(SimpleNamespace(), tmp_path, COURSE, None, shown.append)
    movies.pending = Cut(b"newer", {"frames": 2})

    movies._present(Cut(b"older", {"frames": 1}))

    assert shown == []


def test_a_movie_without_an_upload_is_shown_locally(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(screen, "render_video", render)
    shown: list[dict] = []

    MovieScreen(SimpleNamespace(), tmp_path, COURSE, None, shown.append)._present(Cut(b"bk2", {"frames": 1}))

    assert shown[0]["frames"] == 1


def test_a_movie_that_does_not_render_is_skipped(tmp_path: Path, monkeypatch):
    def fail(paths: object, recording: Path) -> Path:
        raise subprocess.CalledProcessError(1, ["python"], "", "ffmpeg failed")

    monkeypatch.setattr(screen, "render_video", fail)
    shown: list[dict] = []

    MovieScreen(SimpleNamespace(), tmp_path, COURSE, None, shown.append)._present(Cut(b"bk2", {"frames": 1}))

    assert shown == []
