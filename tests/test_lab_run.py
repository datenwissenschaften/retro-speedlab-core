import json
import subprocess
import time
from pathlib import Path

from datenwissenschaften.lab.activity import Action, log_events, recent_actions, shown_file
from datenwissenschaften.lab.branch import branch_changes, shown_subject
from datenwissenschaften.training import lab_run as lab_run_module
from datenwissenschaften.training.lab_run import LabRun


def write_marker(marker: Path, until: float, agents: dict[str, dict[str, str]], base: str, phase: str) -> None:
    lab = {"until": until, "phase": phase, "base": base, "agents": agents, "duel": None}
    marker.write_text(json.dumps(lab), encoding="utf-8")


def write_log(log: Path, events: list[object]) -> None:
    lines = [json.dumps(event) for event in events]
    log.write_text("\n".join(["Error: rate limited", *lines, '{"type": "assist']), encoding="utf-8")


def git(worktree: Path, *arguments: str) -> str:
    command = ["git", "-C", str(worktree), "-c", "user.name=lab", "-c", "user.email=lab@example.com", *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def test_training_pauses_only_while_a_live_lab_run_judges_or_deploys(tmp_path: Path):
    marker = tmp_path / ".lab-run"
    run = LabRun(marker)

    assert not run.paused()
    write_marker(marker, time.time() - 60, {}, "HEAD", "judging")
    assert not run.paused()
    write_marker(marker, time.time() + 600, {}, "HEAD", "working")
    assert not run.paused()
    write_marker(marker, time.time() + 600, {}, "HEAD", "judging")
    assert run.paused()


def test_training_runs_on_while_the_agents_work_and_checks_the_lab_once_per_poll(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    write_marker(marker, time.time() + 600, {}, "HEAD", "working")
    published = []
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values, replace: published.append(values))
    run = LabRun(marker)

    assert (run.wait(), run.wait()) == (None, None)
    assert [values["phase"] for values in published] == ["working"]


def test_training_waits_until_the_lab_run_removes_its_marker(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    write_marker(marker, time.time() + 600, {}, "HEAD", "judging")
    published, sleeps = [], []
    monkeypatch.setattr(
        lab_run_module, "publish_metadata", lambda section, values, replace: published.append(values["active"])
    )
    monkeypatch.setattr(lab_run_module, "consume_model_reset", lambda: None)

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 2:
            marker.unlink()

    monkeypatch.setattr(lab_run_module.time, "sleep", sleep)

    assert LabRun(marker).wait() is None
    assert (published, len(sleeps)) == ([True, True, False], 2)


def test_a_model_reset_during_the_pause_is_handed_back(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    write_marker(marker, time.time() + 600, {}, "HEAD", "decided")
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values, replace: None)
    monkeypatch.setattr(lab_run_module, "consume_model_reset", lambda: "reset")

    assert LabRun(marker).wait() == "reset"


def test_without_a_lab_run_the_stream_is_told_training_runs(tmp_path: Path, monkeypatch):
    published = []
    monkeypatch.setattr(lab_run_module, "publish_metadata", lambda section, values, replace: published.append(values))

    assert LabRun(tmp_path / ".lab-run").wait() is None
    assert published == [{"active": False, "until": None}]


def test_only_source_files_inside_the_branch_are_named(tmp_path: Path):
    assert shown_file(f"{tmp_path.as_posix()}/src/game/reward.py", tmp_path) == "src/game/reward.py"
    assert shown_file("tests/test_reward.py", tmp_path) == "tests/test_reward.py"
    assert shown_file(f"{tmp_path.as_posix()}/config.yaml", tmp_path) is None
    assert shown_file(f"{tmp_path.as_posix()}/src/.env", tmp_path) is None
    assert shown_file(f"{tmp_path.as_posix()}/src/../config.yaml", tmp_path) is None
    assert shown_file("/workspace/.codex/auth.json", tmp_path) is None
    assert shown_file("/app/src/game.py", tmp_path) is None


def test_claude_activity_shows_kinds_and_safe_files_only(tmp_path: Path):
    log = tmp_path / "claude.jsonl"
    content = [
        {"type": "text", "text": "The token is sk-secret"},
        {"type": "tool_use", "id": "1", "name": "Bash", "input": {"command": "cat config.yaml"}},
        {"type": "tool_use", "id": "2", "name": "Edit", "input": {"file_path": f"{tmp_path.as_posix()}/src/a.py"}},
        {"type": "tool_use", "id": "3", "name": "Read", "input": {"file_path": f"{tmp_path.as_posix()}/config.yaml"}},
    ]
    write_log(
        log,
        [
            {"type": "system"},
            {"type": "assistant", "message": {"content": content}, "parent_tool_use_id": None},
            {"type": "assistant"},
        ],
    )

    assert recent_actions("claude", log_events(log), tmp_path) == [
        Action("thinking", None),
        Action("running", None),
        Action("editing", "src/a.py"),
        Action("reading", None),
    ]


def test_codex_activity_counts_every_item_once(tmp_path: Path):
    log = tmp_path / "codex.jsonl"
    command = {"id": "item_1", "type": "command_execution", "command": "printenv"}
    change = {"id": "item_2", "type": "file_change", "changes": [{"path": f"{tmp_path.as_posix()}/src/b.py"}]}
    write_log(
        log,
        [
            {"type": "thread.started", "thread_id": "x"},
            {"type": "item.started", "item": command},
            {"type": "item.completed", "item": {**command, "aggregated_output": "TOKEN=abc"}},
            {"type": "item.completed", "item": change},
        ],
    )

    assert recent_actions("codex", log_events(log), tmp_path) == [
        Action("running", None),
        Action("editing", "src/b.py"),
    ]


def test_commit_subjects_hide_urls_and_secret_like_words():
    subject = "fix(reward): token ghp_a1B2c3D4e5F6g7H8i9J0k1 from https://x.example/t?k=1 \x1b[31m"

    assert shown_subject(subject) == "fix(reward): token [hidden] from [hidden] [31m"


def test_branch_changes_count_commits_and_lines_since_the_base(tmp_path: Path):
    git(tmp_path, "init", "--quiet")
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "--quiet", "-m", "start")
    base = git(tmp_path, "rev-parse", "HEAD")
    (tmp_path / "a.py").write_text("two\nthree\n", encoding="utf-8")
    git(tmp_path, "commit", "--quiet", "-am", "feat: two lines")

    changes = branch_changes(tmp_path, base)

    assert changes is not None
    assert (changes.commits, changes.subjects, changes.files, changes.added, changes.removed) == (
        1,
        ["feat: two lines"],
        1,
        2,
        1,
    )
    assert branch_changes(tmp_path / "removed", base) is None
