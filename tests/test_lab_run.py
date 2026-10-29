import json
import subprocess
import time
from pathlib import Path

from datenwissenschaften.lab import run as lab_run
from datenwissenschaften.lab.activity import Action, recent_actions, shown_file
from datenwissenschaften.lab.branch import branch_changes, shown_subject


def write_marker(marker: Path, until: float, agents: dict[str, dict[str, str]], base: str, phase: str) -> None:
    lab = {"until": until, "phase": phase, "base": base, "agents": agents, "duel": None}
    marker.write_text(json.dumps(lab), encoding="utf-8")


def write_log(log: Path, events: list[object]) -> None:
    lines = [json.dumps(event) for event in events]
    log.write_text("\n".join(["Error: rate limited", *lines, '{"type": "assist']), encoding="utf-8")


def git(worktree: Path, *arguments: str) -> str:
    command = ["git", "-C", str(worktree), "-c", "user.name=lab", "-c", "user.email=lab@example.com", *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()


def test_without_a_lab_run_the_stream_is_told_none_runs(tmp_path: Path, monkeypatch):
    published = []
    monkeypatch.setattr(lab_run, "publish_metadata", lambda section, values: published.append((section, values)))

    lab_run.publish_lab_run(tmp_path / ".lab-run")

    assert published == [("lab_run", {"active": False, "until": None})]


def test_a_live_lab_run_is_published_with_its_phase_and_agents(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    worktree = tmp_path / "missing"
    write_marker(
        marker, time.time() + 600, {"claude": {"log": "none.jsonl", "worktree": str(worktree)}}, "HEAD", "working"
    )
    published = []
    monkeypatch.setattr(lab_run, "publish_metadata", lambda section, values: published.append(values))

    lab_run.publish_lab_run(marker)

    assert published[0]["phase"] == "working"
    assert published[0]["agents"] == {"claude": {"actions": [], "changes": None}}


def test_an_expired_marker_counts_as_no_lab_run(tmp_path: Path, monkeypatch):
    marker = tmp_path / ".lab-run"
    write_marker(marker, time.time() - 60, {}, "HEAD", "working")
    published = []
    monkeypatch.setattr(lab_run, "publish_metadata", lambda section, values: published.append(values))

    lab_run.publish_lab_run(marker)

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
    write_log(log, [{"type": "system"}, {"type": "assistant", "message": {"content": content}}, {"type": "assistant"}])

    assert recent_actions("claude", log, tmp_path) == [
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

    assert recent_actions("codex", log, tmp_path) == [Action("running", None), Action("editing", "src/b.py")]


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
