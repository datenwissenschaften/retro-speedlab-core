import json
import os
import time
from pathlib import Path

import pytest
from fakes import FakeWrapper, write_config

from datenwissenschaften.tas.stuck import STUCK_FILE, mark_stuck
from datenwissenschaften.workflow import lab as lab_module
from datenwissenschaften.workflow.lab import lab, lab_run_due
from datenwissenschaften.workflow.wrapper import wrapper_class

STALLED_SECONDS_AGO = 31 * 60


def progress(tmp_path: Path, seconds_ago: float) -> None:
    movie = tmp_path / "cache" / "tas" / "progress.msgpack"
    movie.parent.mkdir(parents=True, exist_ok=True)
    movie.write_bytes(b"")
    moment = time.time() - seconds_ago
    os.utime(movie, (moment, moment))


def test_a_wrapper_is_named_by_its_module_and_class():
    assert wrapper_class("fakes:FakeWrapper") is FakeWrapper
    with pytest.raises(ValueError):
        wrapper_class("fakes.FakeWrapper")


def test_a_lab_run_is_due_once_the_search_made_no_progress_for_the_stall_minutes(tmp_path: Path):
    config = str(write_config(tmp_path))

    assert lab_run_due(config)
    progress(tmp_path, STALLED_SECONDS_AGO)
    assert lab_run_due(config)


def test_no_lab_run_while_the_search_progresses(tmp_path: Path):
    config = str(write_config(tmp_path))
    progress(tmp_path, 60)

    assert not lab_run_due(config)


def test_a_stuck_search_lets_the_lab_in_at_once(tmp_path: Path):
    config = str(write_config(tmp_path))
    progress(tmp_path, 60)
    mark_stuck(tmp_path / "cache" / "tas" / STUCK_FILE, "Boss")

    assert lab_run_due(config)


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


def reports(project: Path, *stamps: str) -> None:
    root = project / "agent" / "reports"
    root.mkdir(parents=True, exist_ok=True)
    for stamp in stamps:
        (root / f"{stamp}.md").write_text("report", encoding="utf-8")


def test_the_critic_reviews_every_five_new_reports_when_no_lab_run_is_due(tmp_path: Path, monkeypatch):
    commands = []
    monkeypatch.setattr(lab_module.subprocess, "run", lambda command, check: commands.append(command))
    config = str(write_config(tmp_path))
    progress(tmp_path, 60)
    reports(tmp_path, "2026-10-29T1300", "2026-10-29T1330", "2026-10-29T1400", "2026-10-29T1430")

    lab(config, str(tmp_path))
    reports(tmp_path, "2026-10-29T1500")
    lab(config, str(tmp_path))
    critic = {"at": "2026-10-29T15:20:00+00:00", "runs": 5, "lessons": []}
    (tmp_path / ".lab-critic.json").write_text(json.dumps(critic), encoding="utf-8")
    reports(tmp_path, "2026-10-29T1530", "2026-10-29T1600", "2026-10-29T1630", "2026-10-29T1700")
    lab(config, str(tmp_path))

    assert commands == [[tmp_path / "agent" / "critic.sh"]]
