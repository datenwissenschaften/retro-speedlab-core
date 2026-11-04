import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import msgspec
import typer
from langfuse import Langfuse
from loguru import logger
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import Span, Tracer

from datenwissenschaften.lab.transcript import AGENT, GENERATION, Round, Step, read_transcript
from datenwissenschaften.logger import setup_logging

LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
TRACER_NAME = "retro-speedlab-lab"
TRACE_NAME = "lab run"
CHAIN = "chain"
ERROR_LEVEL = "ERROR"
REJECTED = "rejected"
DEPLOYED = "deployed"
OUTCOMES = frozenset({REJECTED, "notes", DEPLOYED})
BOOLEAN_SCORES = frozenset({"accepted", DEPLOYED})
LOG_LEVEL = "INFO"
MAX_TEXT = 4000
NANOSECONDS = 1_000_000_000
SECONDS_PER_MINUTE = 60
TYPE = "langfuse.observation.type"
INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
METADATA = "langfuse.observation.metadata"
LEVEL = "langfuse.observation.level"
MODEL = "langfuse.observation.model.name"
USAGE = "langfuse.observation.usage_details"
TRACE_NAME_KEY = "langfuse.trace.name"
TRACE_TAGS = "langfuse.trace.tags"
TRACE_METADATA = "langfuse.trace.metadata"


@dataclass(slots=True, frozen=True)
class LabRun:
    game: str
    stamp: str
    base: str
    outcome: str
    commits: int

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown lab run outcome {self.outcome!r}, expected one of {sorted(OUTCOMES)}")


def nanoseconds(moment: datetime) -> int:
    return int(moment.timestamp() * NANOSECONDS)


def encoded(value: Any) -> str:
    return msgspec.json.encode(value).decode()


def clipped(value: Any) -> str:
    return (value if isinstance(value, str) else encoded(value))[:MAX_TEXT]


def exports_every_span(span: ReadableSpan) -> bool:
    return True


def step_attributes(step: Step) -> dict[str, Any]:
    attributes: dict[str, Any] = {TYPE: step.kind, OUTPUT: clipped(step.output)}
    if step.input is not None:
        attributes[INPUT] = clipped(step.input)
    if step.kind == GENERATION:
        attributes[MODEL], attributes[USAGE] = step.name, encoded(step.usage)
    if step.failed:
        attributes[LEVEL] = ERROR_LEVEL
    return attributes


class RoundExporter:
    def __init__(self, tracer: Tracer, steps: list[Step]) -> None:
        self.tracer = tracer
        keys = {step.key for step in steps}
        self.children: dict[str | None, list[Step]] = {}
        for step in steps:
            self.children.setdefault(step.owner if step.owner in keys else None, []).append(step)

    def below(self, key: str | None) -> list[Step]:
        return self.children[key] if key in self.children else []

    def emit(self, step: Step, parent: Span) -> datetime:
        context = trace.set_span_in_context(parent)
        span = self.tracer.start_span(
            step.name, context, start_time=nanoseconds(step.start), attributes=step_attributes(step)
        )
        end = max([step.end, *(self.emit(child, span) for child in self.below(step.key))])
        span.end(end_time=nanoseconds(end))
        return end


def export_run(tracer: Tracer, run: LabRun, rounds: list[Round]) -> str:
    root = tracer.start_span(
        TRACE_NAME,
        start_time=nanoseconds(rounds[0].start),
        attributes={
            TRACE_NAME_KEY: TRACE_NAME,
            TRACE_TAGS: [run.game, run.outcome],
            TRACE_METADATA: encoded({"stamp": run.stamp, "base": run.base, "commits": run.commits}),
            TYPE: AGENT,
            INPUT: encoded({"game": run.game, "base": run.base}),
            OUTPUT: clipped(rounds[-1].result),
        },
    )
    for number, lab_round in enumerate(rounds, 1):
        attributes = {TYPE: CHAIN, OUTPUT: clipped(lab_round.result), METADATA: encoded({"cost_usd": lab_round.cost})}
        context = trace.set_span_in_context(root)
        span = tracer.start_span(
            f"round {number}", context, start_time=nanoseconds(lab_round.start), attributes=attributes
        )
        exporter = RoundExporter(tracer, lab_round.steps)
        for step in exporter.below(None):
            exporter.emit(step, span)
        span.end(end_time=nanoseconds(lab_round.end))
    root.end(end_time=nanoseconds(rounds[-1].end))
    return format(root.get_span_context().trace_id, "032x")


def run_scores(run: LabRun, rounds: list[Round]) -> dict[str, float]:
    steps = [step for lab_round in rounds for step in lab_round.steps]
    return {
        "accepted": float(run.outcome != REJECTED),
        DEPLOYED: float(run.outcome == DEPLOYED),
        "cost_usd": sum(lab_round.cost for lab_round in rounds),
        "commits": float(run.commits),
        "minutes": (rounds[-1].end - rounds[0].start).total_seconds() / SECONDS_PER_MINUTE,
        "turns": float(sum(step.kind == GENERATION for step in steps)),
        "subagents": float(sum(step.kind == AGENT for step in steps)),
        "tool_errors": float(sum(step.failed for step in steps)),
    }


def trace_run(langfuse: Langfuse, tracer: Tracer, run: LabRun, rounds: list[Round]) -> str:
    trace_id = export_run(tracer, run, rounds)
    for name, value in run_scores(run, rounds).items():
        data_type = "BOOLEAN" if name in BOOLEAN_SCORES else "NUMERIC"
        langfuse.create_score(name=name, value=value, trace_id=trace_id, data_type=data_type)
    return trace_id


app = typer.Typer()


@app.command()
def trace_lab_run(log: Path, game: str, stamp: str, base: str, outcome: str, commits: int) -> None:
    setup_logging(LOG_LEVEL)
    run = LabRun(game, stamp, base, outcome, commits)
    missing = [key for key in LANGFUSE_KEYS if key not in os.environ]
    if missing:
        logger.warning(f"The lab run {stamp} is not traced: {', '.join(missing)} not set")
        return
    rounds = read_transcript(log)
    if not rounds:
        raise ValueError(f"the lab log {log} has no finished round")
    provider = TracerProvider()
    langfuse = Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        base_url=os.environ["LANGFUSE_BASE_URL"],
        tracer_provider=provider,
        should_export_span=exports_every_span,
    )
    trace_id = trace_run(langfuse, provider.get_tracer(TRACER_NAME), run, rounds)
    langfuse.shutdown()
    logger.info(f"The lab run {stamp} is traced: {langfuse.get_trace_url(trace_id=trace_id)}")


if __name__ == "__main__":
    app()
