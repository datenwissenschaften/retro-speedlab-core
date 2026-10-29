import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import write_config

from datenwissenschaften.settings import load_config
from datenwissenschaften.workflow import speedrun as speedrun_module


def test_the_stream_relay_runs_only_with_the_ui_and_needs_a_key(tmp_path: Path, monkeypatch):
    started: list[str] = []
    relay = lambda url, key, ui, live: SimpleNamespace(start=lambda: started.append(url))  # noqa: E731
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


def test_the_lab_status_is_published_until_stopped(tmp_path: Path, monkeypatch):
    published: list[Path] = []
    monkeypatch.setattr(speedrun_module, "publish_lab_run", published.append)

    speedrun_module.watch_lab(tmp_path / ".lab-run", SimpleNamespace(wait=lambda seconds: bool(published)))

    assert published == [tmp_path / ".lab-run"]


def test_the_speedrun_task_relays_watches_the_lab_and_runs(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(speedrun_module, "start_relay", lambda settings: calls.append("relay"))
    monkeypatch.setattr(speedrun_module, "speedrun_forever", lambda wrapper, config: calls.append("run"))

    speedrun_module.speedrun("fakes:FakeWrapper", str(write_config(tmp_path)))

    assert calls == ["relay", "run"]
