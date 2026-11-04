from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import msgspec

GENERATION = "generation"
TOOL = "tool"
AGENT = "agent"
AGENT_TOOLS = frozenset({"Agent", "Task"})
USAGE_KEYS = {
    "input_tokens": "input",
    "output_tokens": "output",
    "cache_read_input_tokens": "cache_read_input_tokens",
    "cache_creation_input_tokens": "cache_creation_input_tokens",
}


@dataclass(slots=True)
class Step:
    kind: str
    key: str
    name: str
    owner: str | None
    start: datetime
    end: datetime
    input: Any
    output: str
    usage: dict[str, int]
    failed: bool


@dataclass(slots=True)
class Round:
    start: datetime
    end: datetime
    cost: float
    result: str
    failed: bool
    steps: list[Step]


def moment(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(block["text"] for block in content if block["type"] == "text")


def usage_of(usage: dict[str, Any]) -> dict[str, int]:
    return {name: usage[key] for key, name in USAGE_KEYS.items() if key in usage}


class Reader:
    def __init__(self) -> None:
        self.rounds: list[Round] = []
        self.steps: dict[str, Step] = {}
        self.last: dict[str | None, datetime] = {}
        self.costs: dict[str, float] = {}
        self.now: datetime | None = None
        self.start: datetime | None = None

    def read(self, event: dict[str, Any]) -> None:
        if "timestamp" in event:
            self.now = moment(event["timestamp"])
            self.start = self.now if self.start is None else self.start
        if event["type"] == "assistant":
            self.answer(event["parent_tool_use_id"], event["message"])
        if event["type"] == "user":
            self.results(event["message"]["content"])
        if event["type"] == "result":
            self.close(event)

    def answer(self, owner: str | None, message: dict[str, Any]) -> None:
        now = self.clock()
        if message["id"] not in self.steps:
            start = self.last[owner] if owner in self.last else now
            self.steps[message["id"]] = Step(
                GENERATION, message["id"], message["model"], owner, start, now, None, "", {}, False
            )
        generation = self.steps[message["id"]]
        generation.end, generation.usage = now, usage_of(message["usage"])
        generation.output = "\n".join(filter(None, [generation.output, text_of(message["content"])]))
        for block in message["content"]:
            if block["type"] == "tool_use":
                self.call(owner, block, now)
        self.last[owner] = now

    def call(self, owner: str | None, block: dict[str, Any], now: datetime) -> None:
        agent = block["name"] in AGENT_TOOLS
        name = block["input"]["subagent_type"] if agent and "subagent_type" in block["input"] else block["name"]
        self.steps[block["id"]] = Step(
            AGENT if agent else TOOL, block["id"], name, owner, now, now, block["input"], "", {}, False
        )

    def results(self, content: Any) -> None:
        if isinstance(content, str):
            return
        for block in content:
            if block["type"] == "tool_result" and block["tool_use_id"] in self.steps:
                step = self.steps[block["tool_use_id"]]
                step.end, step.output = self.clock(), text_of(block["content"])
                step.failed = "is_error" in block and bool(block["is_error"])
                self.last[step.owner] = step.end

    def close(self, event: dict[str, Any]) -> None:
        end = self.clock()
        session, total = event["session_id"], event["total_cost_usd"]
        cost = total - (self.costs[session] if session in self.costs else 0.0)
        self.costs[session] = total
        result = event["result"] if "result" in event else ""
        start = self.start if self.start is not None else end
        self.rounds.append(Round(start, end, cost, result, event["is_error"], list(self.steps.values())))
        self.steps, self.last, self.start = {}, {}, None

    def clock(self) -> datetime:
        if self.now is None:
            raise ValueError("the lab log has an event before its first timestamp")
        return self.now


def read_transcript(log: Path) -> list[Round]:
    reader = Reader()
    for line in log.read_bytes().splitlines():
        try:
            event = msgspec.json.decode(line)
        except msgspec.DecodeError:
            continue
        if isinstance(event, dict) and "type" in event:
            reader.read(cast(dict[str, Any], event))
    return reader.rounds
