import json
import time
from datetime import UTC, datetime
from pathlib import Path

import msgspec
import pytest
from fakes import FakeWrapper, write_config

from datenwissenschaften.speedrun.journal import JOURNAL_FILE, Journal, JournalStore
from datenwissenschaften.speedrun.routes import ROUTES_FILE
from datenwissenschaften.speedrun.run import BACKTRACK_SEARCHES
from datenwissenschaften.speedrun.stuck import STUCK_FILE, mark_stuck
from datenwissenschaften.workflow import lab as lab_module
from datenwissenschaften.workflow.lab import lab, lab_run_due
from datenwissenschaften.workflow.wrapper import wrapper_class

STALLED_SECONDS_AGO = 31 * 60


def progress(tmp_path: Path, seconds_ago: float) -> None:
    path = tmp_path / "cache" / "tas" / JOURNAL_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    moment = datetime.fromtimestamp(time.time() - seconds_ago, UTC).isoformat(timespec="seconds")
    path.write_bytes(msgspec.json.encode(Journal([], [], [], {}, {}, moment)))


def searching(tmp_path: Path) -> None:
    (tmp_path / "cache" / "tas" / "live").mkdir(parents=True, exist_ok=True)


def test_a_wrapper_is_named_by_its_module_and_class():
    assert wrapper_class("fakes:FakeWrapper") is FakeWrapper
    with pytest.raises(ValueError):
        wrapper_class("fakes.FakeWrapper")


def test_a_lab_run_is_due_once_the_search_made_no_progress_for_the_stall_minutes(tmp_path: Path):
    config = str(write_config(tmp_path))
    searching(tmp_path)

    assert lab_run_due(config)
    progress(tmp_path, STALLED_SECONDS_AGO)
    assert lab_run_due(config)


def test_no_lab_run_while_the_search_progresses(tmp_path: Path):
    config = str(write_config(tmp_path))
    progress(tmp_path, 60)

    assert not lab_run_due(config)


def test_a_stuck_level_alone_calls_no_lab_run_while_another_level_stores_pieces(tmp_path: Path):
    config = str(write_config(tmp_path))
    searching(tmp_path)
    progress(tmp_path, STALLED_SECONDS_AGO)
    for _ in range(3 * BACKTRACK_SEARCHES):
        mark_stuck(tmp_path / "cache" / "tas" / STUCK_FILE, "Boss")
    JournalStore(tmp_path / "cache" / "tas" / JOURNAL_FILE).event("found", "Peak", 300)

    assert not lab_run_due(config)


def test_checkpoints_and_stored_routes_are_no_progress_that_holds_off_the_lab(tmp_path: Path):
    config = str(write_config(tmp_path))
    searching(tmp_path)
    journal = JournalStore(tmp_path / "cache" / "tas" / JOURNAL_FILE)
    journal.event("checkpoint", "Boss", 420)
    journal.event("stuck", "Boss", None)
    (tmp_path / "cache" / "tas" / ROUTES_FILE).write_bytes(b"")

    assert lab_run_due(config)
    journal.event("found", "Boss", 420)
    assert not lab_run_due(config)


def test_finding_the_way_into_a_state_again_is_no_progress_that_holds_off_the_lab(tmp_path: Path):
    config = str(write_config(tmp_path))
    searching(tmp_path)
    path = tmp_path / "cache" / "tas" / JOURNAL_FILE
    JournalStore(path).event("found", "Boss", 900)
    progress_at = datetime.fromtimestamp(time.time() - STALLED_SECONDS_AGO, UTC).isoformat(timespec="seconds")
    journal = msgspec.json.decode(path.read_bytes(), type=Journal)
    path.write_bytes(msgspec.json.encode(msgspec.structs.replace(journal, progressed=progress_at)))

    JournalStore(path).event("found", "Boss", 1100)
    assert lab_run_due(config)
    JournalStore(path).event("found", "Peak", 300)
    assert not lab_run_due(config)


def test_no_lab_run_while_the_search_is_not_running(tmp_path: Path):
    config = str(write_config(tmp_path))
    mark_stuck(tmp_path / "cache" / "tas" / STUCK_FILE, "Boss")

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
    searching(tmp_path)

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
