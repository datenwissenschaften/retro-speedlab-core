import json
import os
import time
from pathlib import Path

import pytest
from fakes import FakeWrapper, write_config

from datenwissenschaften.environment import level_drill
from datenwissenschaften.workflow import lab as lab_module
from datenwissenschaften.workflow import training as training_module
from datenwissenschaften.workflow.lab import lab, lab_run_due
from datenwissenschaften.workflow.wrapper import wrapper_class

STALLED_SECONDS_AGO = 31 * 60


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


def test_a_session_task_trains_one_session_of_the_named_wrapper(tmp_path: Path, monkeypatch):
    sessions = []

    class Trainer:
        def __init__(self, wrapper_cls: type, config_path: Path) -> None:
            self.started = (wrapper_cls, config_path)

        def train_session(self) -> None:
            sessions.append(self.started)

    monkeypatch.setattr(training_module, "LayaTrainer", Trainer)

    config = write_config(tmp_path)

    training_module.train_session("fakes:FakeWrapper", str(config))

    assert sessions == [(FakeWrapper, config)]


def test_training_trains_one_session_after_another(monkeypatch):
    sessions: list[tuple[str, str]] = []

    def session(wrapper: str, config: str) -> None:
        sessions.append((wrapper, config))
        if len(sessions) > 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(training_module, "train_session", session)

    with pytest.raises(KeyboardInterrupt):
        training_module.training("game:Wrapper", "config.yaml")

    assert sessions == [("game:Wrapper", "config.yaml")] * 2


def test_a_lab_run_is_due_once_training_made_no_progress_for_the_stall_minutes(tmp_path: Path):
    config = str(write_config(tmp_path))

    assert lab_run_due(config)
    progress(tmp_path, STALLED_SECONDS_AGO)
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


def drill_files(tmp_path: Path, endings: int, tries: int) -> Path:
    root = tmp_path / "cache" / "curriculum" / "Level1"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"Level 1{level_drill.ENDINGS_SUFFIX}").write_text(json.dumps(["Boss"] * endings), encoding="utf-8")
    if tries:
        (root / "Level 1.drill.json").write_text(json.dumps([False] * tries), encoding="utf-8")
    return root


def test_no_lab_run_while_the_engine_collects_endings_or_drills_a_level(tmp_path: Path):
    config = str(write_config(tmp_path))
    drill_files(tmp_path, level_drill.MIN_ENDINGS - 1, 0)
    assert not lab_run_due(config)

    drill_files(tmp_path, level_drill.MIN_ENDINGS, level_drill.DRILL_WINDOW - 1)
    assert not lab_run_due(config)


def test_a_drill_that_keeps_failing_lets_the_lab_in(tmp_path: Path):
    config = str(write_config(tmp_path))
    drill_files(tmp_path, level_drill.MIN_ENDINGS, level_drill.DRILL_WINDOW)

    assert lab_run_due(config)


def test_a_passed_drill_counts_as_progress(tmp_path: Path):
    config = str(write_config(tmp_path))
    (drill_files(tmp_path, 0, 0) / f"Level 1{level_drill.DRILLED_SUFFIX}").touch()

    assert not lab_run_due(config)
