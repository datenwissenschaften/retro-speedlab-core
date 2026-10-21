import os
import time
from pathlib import Path

import pytest
from fakes import FakeWrapper, write_config

from datenwissenschaften.workflow import lab as lab_module
from datenwissenschaften.workflow import training as training_module
from datenwissenschaften.workflow.lab import lab, lab_run_due
from datenwissenschaften.workflow.training import wrapper_class

HOURS_AGO = 3 * 60 * 60


def progress(tmp_path: Path, seconds_ago: float) -> None:
    successes = tmp_path / "cache" / "curriculum" / "Level1" / "Survive.successes"
    successes.parent.mkdir(parents=True)
    successes.write_text("1", encoding="utf-8")
    moment = time.time() - seconds_ago
    os.utime(successes, (moment, moment))


def test_a_wrapper_is_named_by_its_module_and_class():
    assert wrapper_class("fakes:FakeWrapper") is FakeWrapper
    with pytest.raises(ValueError):
        wrapper_class("fakes.FakeWrapper")


def test_training_runs_one_session_after_another(tmp_path: Path, monkeypatch):
    sessions = []

    class Trainer:
        def __init__(self, wrapper_cls: type, config_path: Path) -> None:
            sessions.append((wrapper_cls, config_path))

        def train_session(self) -> None:
            if len(sessions) == 3:
                raise KeyboardInterrupt

    monkeypatch.setattr(training_module, "LayaTrainer", Trainer)

    with pytest.raises(KeyboardInterrupt):
        training_module.training("fakes:FakeWrapper", str(tmp_path / "config.yaml"))

    assert sessions == [(FakeWrapper, tmp_path / "config.yaml")] * 3


def test_a_lab_run_is_due_once_training_made_no_progress_for_the_stall_hours(tmp_path: Path):
    config = str(write_config(tmp_path))

    assert lab_run_due(config)
    progress(tmp_path, HOURS_AGO)
    assert lab_run_due(config)


def test_no_lab_run_while_training_progresses(tmp_path: Path):
    config = str(write_config(tmp_path))
    progress(tmp_path, 60)

    assert not lab_run_due(config)


def test_the_end_of_the_last_lab_run_counts_as_progress(tmp_path: Path):
    config = str(write_config(tmp_path))
    (tmp_path / lab_module.LAST_LAB_RUN).touch()

    assert not lab_run_due(config)


def test_no_second_lab_run_while_one_is_running(tmp_path: Path):
    config = str(write_config(tmp_path))
    (tmp_path / "lab-run").write_text(f"{time.time() + 600:.0f}\n", encoding="utf-8")

    assert not lab_run_due(config)


def test_the_lab_runs_the_projects_daily_agent_only_when_due(tmp_path: Path, monkeypatch):
    commands = []
    monkeypatch.setattr(lab_module.subprocess, "run", lambda command, check: commands.append(command))
    config = str(write_config(tmp_path))

    lab(config, str(tmp_path))
    progress(tmp_path, 60)
    lab(config, str(tmp_path))

    assert commands == [[tmp_path / "agent" / "daily.sh"]]
