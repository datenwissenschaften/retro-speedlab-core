import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from datenwissenschaften.lab.activity import THINKING, WORKING, claude_action
from datenwissenschaften.lab.branch import shown_subject
from datenwissenschaften.lab.roles import SAFE_NAME, agent_roles

LEAD = "lead"
LEAD_ROLE = "Plans the run and decides what ships"
WAITING = "waiting"
DONE = "done"
DELEGATING_TOOLS = frozenset({"Agent", "Task"})
SHOWN_HANDOFFS = 10
TASK_WORDS = 5
SAFE_SITE = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)+")
SITE_PREFIX = "www."


@dataclass(slots=True)
class Member:
    name: str
    role: str | None
    state: str
    step: str | None
    since: str | None


def shown_site(url: str) -> str | None:
    host = urlsplit(url).hostname
    if host is None or not SAFE_SITE.fullmatch(host):
        return None
    return host.removeprefix(SITE_PREFIX)


def shown_task(description: str) -> str:
    return " ".join(shown_subject(description).split()[:TASK_WORDS])


def step(tool_use: dict[str, Any], worktree: Path) -> str:
    action = claude_action(tool_use, worktree)
    site = shown_site(tool_use["input"]["url"]) if "url" in tool_use["input"] else None
    target = action.file if action.file is not None else site
    return action.kind.capitalize() if target is None else f"{action.kind.capitalize()} {target}"


class Team:
    def __init__(self, roles: dict[str, str], worktree: Path) -> None:
        self.roles = roles
        self.worktree = worktree
        self.members = {LEAD: Member(LEAD, LEAD_ROLE, WORKING, None, None)}
        self.calls: dict[str, str] = {}
        self.running: dict[str, str] = {}
        self.handoffs: list[dict[str, str]] = []
        self.now: str | None = None

    def read(self, event: dict[str, Any]) -> None:
        if "timestamp" in event:
            self.now = event["timestamp"]
        if event["type"] == "result":
            self.members[LEAD].state = DONE
        if event["type"] == "system" and event["subtype"] == "task_notification":
            self.finish(event["tool_use_id"])
        if event["type"] == "assistant":
            self.act(self.owner(event), event["message"]["content"])

    def owner(self, event: dict[str, Any]) -> str:
        parent = event["parent_tool_use_id"]
        return LEAD if parent is None else self.calls[parent]

    def act(self, name: str, blocks: list[dict[str, Any]]) -> None:
        member = self.members[name]
        member.state, member.since = WORKING, self.now
        tool_uses = [block for block in blocks if block["type"] == "tool_use"]
        for tool_use in tool_uses:
            if tool_use["name"] in DELEGATING_TOOLS:
                self.delegate(member, tool_use["id"], tool_use["input"])
                continue
            member.step = step(tool_use, self.worktree)
        if member.step is None:
            member.step = THINKING.capitalize()

    def delegate(self, sender: Member, call: str, arguments: dict[str, Any]) -> None:
        name = arguments["subagent_type"]
        if not SAFE_NAME.fullmatch(name):
            return
        sender.step = f"Delegating to {name}"
        self.calls[call] = self.running[call] = name
        role = self.roles[name] if name in self.roles else None
        self.members[name] = Member(name, role, WORKING, None, self.now)
        handoff = {"at": self.now, "from": sender.name, "to": name, "task": shown_task(arguments["description"])}
        self.handoffs = [*self.handoffs, handoff][-SHOWN_HANDOFFS:]

    def finish(self, call: str) -> None:
        if call not in self.running:
            return
        name = self.running.pop(call)
        if name not in self.running.values():
            self.members[name].state, self.members[name].since = DONE, self.now

    def lineup(self) -> list[Member]:
        waiting = [Member(name, role, WAITING, None, None) for name, role in self.roles.items()]
        return [*self.members.values(), *(member for member in waiting if member.name not in self.members)]


def team_status(events: list[Any], worktree: Path) -> tuple[list[Member], list[dict[str, str]]]:
    team = Team(agent_roles(worktree), worktree)
    for event in events:
        try:
            team.read(event)
        except (KeyError, TypeError, AttributeError):
            continue
    return team.lineup(), team.handoffs
