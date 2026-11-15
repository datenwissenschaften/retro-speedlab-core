from datetime import UTC, datetime
from importlib import import_module
from typing import Any, Protocol

from datenwissenschaften.settings import EventSettings

ATTEMPTS = "runs.attempts"
STEPS = "runs.steps"
RECORDS = "runs.records"
POLICY_ROUNDS = "policy.rounds"
LAB_EVENTS = "lab.events"


class Events(Protocol):
    def publish(self, topic: str, key: str, payload: dict[str, Any]) -> None: ...


class Unpublished:
    def publish(self, topic: str, key: str, payload: dict[str, Any]) -> None:
        return None


def load_events(settings: EventSettings) -> Events:
    if settings.sink is None:
        return Unpublished()
    module, name = settings.sink.split(":")
    sink: Events = getattr(import_module(module), name)(settings.options)
    return sink


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")
