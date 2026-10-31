from pathlib import Path
from types import SimpleNamespace

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
