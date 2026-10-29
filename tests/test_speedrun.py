import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import write_config

from datenwissenschaften.settings import load_config
from datenwissenschaften.tas import run
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.workflow import speedrun as speedrun_module

SLOW = [Segment("Boss", [[1, 0]] * 6, b"", False)]
FAST = [Segment("Boss", [[1, 0]] * 3, b"", False)]


class OnePass(Exception):
    pass


class NoPool:
    def __enter__(self) -> "NoPool":
        return self

    def __exit__(self, *exception: object) -> None:
        pass


def test_the_stream_relay_runs_only_with_the_ui_and_needs_a_key(tmp_path: Path, monkeypatch):
    started: list[str] = []
    relay = lambda url, key, ui, root: SimpleNamespace(start=lambda: started.append(url))  # noqa: E731
    monkeypatch.setattr(speedrun_module, "LiveRelay", relay)
    settings = load_config(write_config(tmp_path))
    with_ui = dataclasses.replace(settings, ui=dataclasses.replace(settings.ui, enabled=True))

    speedrun_module.start_relay(settings)
    with pytest.raises(RuntimeError):
        speedrun_module.start_relay(with_ui)
    speedrun_module.start_relay(
        dataclasses.replace(with_ui, upload=dataclasses.replace(settings.upload, api_key="key"))
    )

    assert started == ["https://upload.test"]


def test_the_lab_status_is_published_until_stopped():
    published: list[int] = []

    speedrun_module.watch_lab(
        SimpleNamespace(publish=lambda: published.append(1)), SimpleNamespace(wait=lambda seconds: bool(published))
    )

    assert published == [1]


def test_the_speedrun_extends_the_movie_then_publishes_every_faster_one(tmp_path: Path, monkeypatch):
    published: list[int] = []

    def improve(segments: list[Segment]) -> list[Segment]:
        if len(published) == 2:
            raise OnePass
        return FAST

    tas = SimpleNamespace(
        output=tmp_path,
        settings=load_config(write_config(tmp_path)),
        extend=lambda segments, until: SLOW,
        improve=improve,
    )
    monkeypatch.setattr(run, "tas_run", lambda wrapper, config, pool: tas)
    monkeypatch.setattr(run, "worker_pool", lambda workers: NoPool())
    monkeypatch.setattr(run, "publish", lambda settings, segments: published.append(run.frames(segments)))

    with pytest.raises(OnePass):
        run.speedrun_forever("fakes:FakeWrapper", write_config(tmp_path))

    assert published == [6, 3]


def test_the_speedrun_task_relays_watches_the_lab_and_runs(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(speedrun_module, "start_relay", lambda settings: calls.append("relay"))
    monkeypatch.setattr(speedrun_module, "speedrun_forever", lambda wrapper, config: calls.append("run"))

    speedrun_module.speedrun("fakes:FakeWrapper", str(write_config(tmp_path)))

    assert calls == ["relay", "run"]
