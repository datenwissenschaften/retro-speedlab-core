import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import msgspec

RECENT_ACTIONS = 6
SHOWN_DIRECTORIES = frozenset({"src", "tests", "integrations"})
SAFE_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
THINKING = "thinking"
WORKING = "working"
CLAUDE_TOOLS = {
    "Bash": "running",
    "Monitor": "running",
    "Read": "reading",
    "Glob": "searching",
    "Grep": "searching",
    "Edit": "editing",
    "Write": "editing",
    "WebSearch": "researching",
    "WebFetch": "researching",
    "Agent": "delegating",
    "Task": "delegating",
}
CLAUDE_TEXT = frozenset({"text", "thinking"})
CODEX_ITEMS = {
    "command_execution": "running",
    "mcp_tool_call": "running",
    "file_change": "editing",
    "web_search": "researching",
    "reasoning": THINKING,
    "agent_message": THINKING,
}
CODEX_EVENTS = frozenset({"item.started", "item.completed"})


@dataclass(slots=True, frozen=True)
class Action:
    kind: str
    file: str | None


def shown_file(path: str, worktree: Path) -> str | None:
    candidate = PurePosixPath(path)
    root = PurePosixPath(worktree.as_posix())
    if candidate.is_absolute() and not candidate.is_relative_to(root):
        return None
    parts = candidate.relative_to(root).parts if candidate.is_absolute() else candidate.parts
    if not parts or parts[0] not in SHOWN_DIRECTORIES:
        return None
    if not all(SAFE_PART.fullmatch(part) for part in parts):
        return None
    return "/".join(parts)


def log_events(log: Path) -> list[Any]:
    if not log.is_file():
        return []
    events = []
    for line in log.read_bytes().splitlines():
        try:
            events.append(msgspec.json.decode(line))
        except msgspec.DecodeError:
            continue
    return events


def claude_action(block: dict[str, Any], worktree: Path) -> Action | None:
    if block["type"] in CLAUDE_TEXT:
        return Action(THINKING, None)
    if block["type"] != "tool_use":
        return None
    name, arguments = block["name"], block["input"]
    kind = CLAUDE_TOOLS[name] if name in CLAUDE_TOOLS else WORKING
    file = shown_file(arguments["file_path"], worktree) if "file_path" in arguments else None
    return Action(kind, file)


def lead_event(event: dict[str, Any]) -> bool:
    return event["type"] == "assistant" and event["parent_tool_use_id"] is None


def claude_actions(key: int, event: dict[str, Any], worktree: Path) -> dict[str, Action]:
    if not lead_event(event):
        return {}
    blocks = enumerate(event["message"]["content"])
    actions = {f"{key}:{index}": claude_action(block, worktree) for index, block in blocks}
    return {key: action for key, action in actions.items() if action is not None}


def codex_actions(key: int, event: dict[str, Any], worktree: Path) -> dict[str, Action]:
    if event["type"] not in CODEX_EVENTS:
        return {}
    item = event["item"]
    kind = CODEX_ITEMS[item["type"]] if item["type"] in CODEX_ITEMS else WORKING
    changes = item["changes"] if item["type"] == "file_change" else []
    files = [file for change in changes if (file := shown_file(change["path"], worktree)) is not None]
    return {item["id"]: Action(kind, files[0] if files else None)}


PARSERS = {"claude": claude_actions, "codex": codex_actions}


def recent_actions(agent: str, events: list[Any], worktree: Path) -> list[Action]:
    parse = PARSERS[agent]
    actions: dict[str, Action] = {}
    for key, event in enumerate(events):
        try:
            actions.update(parse(key, event, worktree))
        except (KeyError, TypeError, AttributeError):
            continue
    return list(actions.values())[-RECENT_ACTIONS:]
