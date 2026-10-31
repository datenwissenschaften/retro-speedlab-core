import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from datenwissenschaften.reference import fetch, paths, tasvideos

PUBLICATION = 1694


def publication_with(monkeypatch: pytest.MonkeyPatch, urls: list[str]) -> None:
    answer = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"urls": urls})
    monkeypatch.setattr(tasvideos.httpx, "get", lambda url, timeout: answer)


def test_the_reference_is_the_video_the_publication_links(monkeypatch: pytest.MonkeyPatch):
    publication_with(monkeypatch, ["https://www.youtube.com/watch?v=tas", "https://archive.org/download/tas/run.mkv"])

    assert tasvideos.video_url(PUBLICATION) == "https://archive.org/download/tas/run.mkv"


def test_a_publication_without_a_video_file_fails(monkeypatch: pytest.MonkeyPatch):
    publication_with(monkeypatch, ["https://www.youtube.com/watch?v=tas"])

    with pytest.raises(ValueError, match="links no"):
        tasvideos.video_url(PUBLICATION)


def test_a_reference_with_frames_is_not_fetched_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = SimpleNamespace(paths=SimpleNamespace(cache_dir=tmp_path))
    (tmp_path / paths.REFERENCE_DIR / paths.FRAMES_DIR).mkdir(parents=True)
    monkeypatch.setattr(fetch, "video_url", lambda publication: pytest.fail("fetched again"))

    fetch.ensure_reference(settings)


class Answer:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def iter_bytes(self) -> list[bytes]:
        return [self.content]

    def __enter__(self) -> "Answer":
        return self

    def __exit__(self, *exception: object) -> None:
        pass


def movie_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("run.fm2", "|0|........|\n")
    return buffer.getvalue()


def test_a_missing_reference_is_downloaded_with_its_movie_and_split_into_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "source.avi"
    video = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 8, (32, 32))
    for _ in range(8):
        video.write(np.zeros((32, 32, 3), np.uint8))
    video.release()
    monkeypatch.setattr(fetch, "video_url", lambda publication: "https://archive.org/download/tas/run.avi")
    monkeypatch.setattr(
        tasvideos.httpx, "stream", lambda method, url, follow_redirects, timeout: Answer(source.read_bytes())
    )
    monkeypatch.setattr(tasvideos.httpx, "get", lambda url, follow_redirects, timeout: Answer(movie_zip()))
    settings = SimpleNamespace(
        paths=SimpleNamespace(cache_dir=tmp_path), speedrun=SimpleNamespace(game=SimpleNamespace(tasvideos=PUBLICATION))
    )

    fetch.ensure_reference(settings)

    reference = tmp_path / paths.REFERENCE_DIR
    assert paths.video_path(settings) == reference / "video.avi"
    assert (reference / paths.MOVIE_DIR / "run.fm2").is_file()
    assert sorted(frame.name for frame in (reference / paths.FRAMES_DIR).glob("*.png"))[:2] == [
        "0000.00s.png",
        "0000.25s.png",
    ]


def test_a_reference_without_one_video_fails(tmp_path: Path):
    settings = SimpleNamespace(paths=SimpleNamespace(cache_dir=tmp_path))

    with pytest.raises(FileNotFoundError, match="not one"):
        paths.video_path(settings)
