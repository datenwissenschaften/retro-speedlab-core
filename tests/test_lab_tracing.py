import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from datenwissenschaften.lab import tracing
from datenwissenschaften.lab.tracing import TYPE, LabRun, export_run, run_scores, trace_lab_run
from datenwissenschaften.lab.transcript import AGENT, GENERATION, TOOL, read_transcript

REAL_LOG = Path(__file__).with_name("claude_team.jsonl")
RUN = LabRun("game", "2026-11-04T1000", "abc1234", "claude", 2)


def message(identifier: str, stamp: str, *content: dict[str, object]) -> dict[str, object]:
    body = {"id": identifier, "model": "claude-opus-5-5", "usage": {"input_tokens": 3, "output_tokens": 5}}
    return {
        "type": "assistant",
        "parent_tool_use_id": None,
        "timestamp": stamp,
        "message": {**body, "content": list(content)},
    }


def result(cost: float) -> dict[str, object]:
    return {"type": "result", "session_id": "s", "total_cost_usd": cost, "is_error": False, "result": "done"}


def two_rounds(tmp_path: Path) -> Path:
    events = [
        message("m1", "2026-11-04T10:00:00.000Z", {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}),
        {
            "type": "user",
            "timestamp": "2026-11-04T10:00:05.000Z",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "boom", "is_error": True},
                ]
            },
        },
        result(1.5),
        message("m2", "2026-11-04T10:01:00.000Z", {"type": "text", "text": "again"}),
        result(2.0),
    ]
    log = tmp_path / "lab.jsonl"
    log.write_text("not json\n" + "\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return log


def spans_of(rounds: list) -> list:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    export_run(provider.get_tracer("test"), RUN, rounds)
    return list(memory.get_finished_spans())


def test_rounds_cost_what_the_session_added(tmp_path: Path) -> None:
    rounds = read_transcript(two_rounds(tmp_path))
    assert [lab_round.cost for lab_round in rounds] == [1.5, 0.5]
    assert [step.kind for step in rounds[0].steps] == [GENERATION, TOOL]
    assert rounds[0].steps[1].failed and rounds[0].steps[1].output == "boom"


def test_scores_measure_the_run(tmp_path: Path) -> None:
    scores = run_scores(RUN, read_transcript(two_rounds(tmp_path)))
    assert scores["accepted"] == 1.0 and scores["cost_usd"] == 2.0 and scores["tool_errors"] == 1.0
    assert scores["turns"] == 2.0 and scores["commits"] == 2.0 and scores["minutes"] == 1.0


def test_subagent_work_is_nested_under_its_agent() -> None:
    spans = spans_of(read_transcript(REAL_LOG))
    agent = next(span for span in spans if span.name == "reader")
    nested = [span for span in spans if span.parent is not None and span.parent.span_id == agent.context.span_id]
    assert agent.attributes[TYPE] == AGENT
    assert {span.name for span in nested} == {"claude-haiku-4-5-20251001", "Read"}
    assert all(span.end_time >= span.start_time for span in spans)


def test_a_run_without_langfuse_keys_is_not_traced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in tracing.LANGFUSE_KEYS:
        monkeypatch.delenv(key, raising=False)
    trace_lab_run(two_rounds(tmp_path), "game", "stamp", "base", "none", 0)
