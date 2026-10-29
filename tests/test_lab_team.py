import json
import time
from pathlib import Path

from datenwissenschaften.lab.activity import Action, log_events, recent_actions
from datenwissenschaften.lab.marker import LabMarker
from datenwissenschaften.lab.roles import agent_roles
from datenwissenschaften.lab.status import lab_status
from datenwissenschaften.lab.team import Member, shown_site, team_status
from datenwissenschaften.training import lab_run as lab_run_module
from datenwissenschaften.training.lab_run import LabRun

REAL_LOG = Path(__file__).with_name("claude_team.jsonl")
LATER = "2026-10-29T15:20:00.000Z"


def define_agent(worktree: Path, name: str, description: str) -> None:
    agents = worktree / ".claude" / "agents"
    agents.mkdir(parents=True, exist_ok=True)
    (agents / f"{name}.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nBody\n", encoding="utf-8")


def real_log(worktree: Path) -> Path:
    log = worktree / "claude.jsonl"
    log.write_text(REAL_LOG.read_text(encoding="utf-8").replace("{worktree}", worktree.as_posix()), encoding="utf-8")
    return log


def assistant(parent: str | None, *content: dict[str, object]) -> dict[str, object]:
    message = {"content": list(content)}
    return {"type": "assistant", "message": message, "parent_tool_use_id": parent, "timestamp": LATER}


def delegate(call: str, name: str, description: str, background: bool) -> dict[str, object]:
    arguments = {"description": description, "subagent_type": name, "prompt": "secret", "run_in_background": background}
    return {"type": "tool_use", "id": call, "name": "Agent", "input": arguments}


def test_a_real_claude_log_shows_the_lead_and_its_subagent(tmp_path: Path):
    define_agent(tmp_path, "reader", "Reads one source file. Use for any file reading task.")
    define_agent(tmp_path, "scout", "Measures RAM in the emulator. Use for probes.")

    team, handoffs = team_status(log_events(real_log(tmp_path)), tmp_path)

    assert team == [
        Member("lead", "Plans the run and decides what ships", "done", "Thinking", "2026-10-29T15:19:13.533Z"),
        Member("reader", "Reads one source file", "done", "Reading src/hello.py", "2026-10-29T15:19:10.345Z"),
        Member("scout", "Measures RAM in the emulator", "waiting", None, None),
    ]
    assert handoffs == [
        {"at": "2026-10-29T15:19:07.522Z", "from": "lead", "to": "reader", "task": "Read src/hello.py and report its"}
    ]


def test_the_lead_actions_leave_out_what_its_subagents_do(tmp_path: Path):
    actions = recent_actions("claude", log_events(real_log(tmp_path)), tmp_path)

    assert actions == [Action("thinking", None), Action("delegating", None)] + [Action("thinking", None)] * 2


def test_parallel_and_background_subagents_stay_working_until_notified(tmp_path: Path):
    fetch = {"type": "tool_use", "id": "f", "name": "WebFetch", "input": {"url": "https://www.tasvideos.org/1694M"}}
    events = [
        assistant(
            None,
            delegate("s", "scout", "Measure the fall", False),
            delegate("r", "researcher", "Find the route", True),
            delegate("x", "bad name!", "Anything", False),
        ),
        assistant("r", fetch),
        assistant("s", {"type": "tool_use", "id": "b", "name": "Bash", "input": {"command": "cat config.yaml"}}),
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "r", "content": "launched"}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "s", "content": "ok"}]}},
        assistant("unknown", {"type": "text", "text": "lost"}),
        {"type": "system", "subtype": "task_notification", "tool_use_id": "missing"},
    ]

    team, handoffs = team_status(events, tmp_path)

    assert [(member.name, member.state, member.step) for member in team] == [
        ("lead", "working", "Delegating to researcher"),
        ("scout", "done", "Running"),
        ("researcher", "working", "Researching tasvideos.org"),
    ]
    assert [handoff["task"] for handoff in handoffs] == ["Measure the fall", "Find the route"]

    team, _ = team_status([*events, {"type": "system", "subtype": "task_notification", "tool_use_id": "r"}], tmp_path)
    assert team[2].state == "done"


def test_handoff_tasks_are_short_and_hide_secret_like_words(tmp_path: Path):
    events = [
        assistant(None, delegate(str(call), "scout", f"Probe ghp_a1B2c3D4e5F6g7H8i9J0k1 now and then {call}", False))
        for call in range(12)
    ]

    _, handoffs = team_status(events, tmp_path)

    assert len(handoffs) == 10
    assert handoffs[-1]["task"] == "Probe [hidden] now and then"


def test_unsafe_sites_and_agent_definitions_are_left_out(tmp_path: Path):
    define_agent(tmp_path, "engineer", "Writes the smallest change and its test. Use for code.")
    define_agent(tmp_path, "evil name", "Anything.")
    (tmp_path / ".claude" / "agents" / "broken.md").write_text("no front matter", encoding="utf-8")

    assert agent_roles(tmp_path) == {"engineer": "Writes the smallest change and its test"}
    assert shown_site("https://user:pass@tasvideos.org/x") == "tasvideos.org"
    assert shown_site("https://localhost:8080/x") is None
    assert shown_site("file:///etc/passwd") is None
    assert shown_site("https://speedrun.com/snake") == "speedrun.com"


def test_the_published_lab_status_names_the_team_and_the_critic(tmp_path: Path, monkeypatch):
    define_agent(tmp_path, "reader", "Reads one source file.")
    log = real_log(tmp_path)
    agents = {"claude": {"log": str(log), "worktree": str(tmp_path)}}
    lab = {"until": time.time() + 600, "phase": "working", "base": "HEAD", "agents": agents, "duel": None}
    marker = tmp_path / ".lab-run"
    marker.write_text(json.dumps(lab), encoding="utf-8")
    critic = {"at": "2026-10-29T15:30:00+00:00", "runs": 5, "lessons": ["Probe RAM first https://x.example/a"]}
    (tmp_path / ".lab-critic.json").write_text(json.dumps(critic), encoding="utf-8")
    published = {}
    monkeypatch.setattr(
        lab_run_module, "publish_metadata", lambda section, values, replace: published.update({section: values})
    )

    LabRun(marker).publish()

    assert [member["name"] for member in published["lab_run"]["team"]] == ["lead", "reader"]
    assert published["lab_run"]["handoffs"][0]["to"] == "reader"
    assert published["critic"] == {**critic, "lessons": ["Probe RAM first [hidden]"]}
    assert lab_status(LabMarker(time.time(), "working", "HEAD", {}, None)).team == []
