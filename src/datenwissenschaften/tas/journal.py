import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from datenwissenschaften.tas.paths import atomic_write

JOURNAL_FILE = "journal.json"
SHOWN_EVENTS = 20
SHOWN_HISTORY = 200
SHOWN_MILESTONES = 50
TENTHS_PER_MINUTE = 600


class Event(msgspec.Struct, frozen=True):
    at: str
    kind: str
    state: str | None
    frames: int | None


class Point(msgspec.Struct, frozen=True):
    at: str
    seconds: float


class Milestone(msgspec.Struct, frozen=True):
    at: str
    kind: str
    text: str


class Journal(msgspec.Struct):
    events: list[Event]
    history: list[Point]
    milestones: list[Milestone]
    firsts: dict[str, float]
    bests: dict[str, float]


@dataclass(slots=True, frozen=True)
class Split:
    level: str
    seconds: float
    level_seconds: float


def clock(seconds: float) -> str:
    tenths = round(seconds * 10)
    return f"{tenths // TENTHS_PER_MINUTE}:{tenths % TENTHS_PER_MINUTE / 10:04.1f}"


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JournalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        empty = Journal([], [], [], {}, {})
        self.journal = msgspec.json.decode(path.read_bytes(), type=Journal) if path.is_file() else empty

    def event(self, kind: str, state: str | None, frames: int | None) -> None:
        with self.lock:
            self.journal.events = [*self.journal.events, Event(now(), kind, state, frames)][-SHOWN_EVENTS:]
            self._save()

    def track(self, splits: list[Split], seconds: float) -> dict[str, float]:
        with self.lock:
            for split in splits:
                self._split(split)
            history = self.journal.history
            if not history or history[-1].seconds != seconds:
                self.journal.history = [*history, Point(now(), seconds)][-SHOWN_HISTORY:]
            self._save()
            return dict(self.journal.firsts)

    def fields(self) -> dict[str, Any]:
        with self.lock:
            journal = self.journal
            return msgspec.to_builtins(
                {"events": journal.events, "history": journal.history, "milestones": journal.milestones}
            )

    def _split(self, split: Split) -> None:
        firsts, bests = self.journal.firsts, self.journal.bests
        if split.level not in firsts:
            firsts[split.level] = bests[split.level] = split.level_seconds
            self._milestone("level", f"{split.level} beaten in {clock(split.level_seconds)}")
            return
        if split.level_seconds < bests[split.level]:
            saved = bests[split.level] - split.level_seconds
            bests[split.level] = split.level_seconds
            self._milestone("best", f"{split.level} in {clock(split.level_seconds)}, {saved:.1f} s faster")

    def _milestone(self, kind: str, text: str) -> None:
        self.journal.milestones = [*self.journal.milestones, Milestone(now(), kind, text)][-SHOWN_MILESTONES:]

    def _save(self) -> None:
        atomic_write(self.path, msgspec.json.encode(self.journal))
