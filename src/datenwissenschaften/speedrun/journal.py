import re
import threading
from collections.abc import Collection
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec

from datenwissenschaften.speedrun.paths import atomic_write

JOURNAL_FILE = "journal.json"
SHOWN_EVENTS = 20
FOUND = "found"
IMPROVED = "improved"
SHOWN_HISTORY = 200
SHOWN_MILESTONES = 50
TENTHS_PER_MINUTE = 600
WORD_START = re.compile(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[0-9])(?=[A-Z])")


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
    progressed: str | None = None
    closest: dict[str, int] = msgspec.field(default_factory=dict[str, int])
    found: list[str] = msgspec.field(default_factory=list[str])


@dataclass(slots=True, frozen=True)
class Split:
    level: str
    seconds: float
    level_seconds: float


def clock(seconds: float) -> str:
    tenths = round(seconds * 10)
    return f"{tenths // TENTHS_PER_MINUTE}:{tenths % TENTHS_PER_MINUTE / 10:04.1f}"


def spoken(state: str) -> str:
    return re.sub(WORD_START, " ", state)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def last_progress(path: Path) -> float:
    if not path.is_file():
        return 0.0
    progressed = msgspec.json.decode(path.read_bytes(), type=Journal).progressed
    return datetime.fromisoformat(progressed).timestamp() if progressed else 0.0


class JournalStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        empty = Journal([], [], [], {}, {})
        self.journal = msgspec.json.decode(path.read_bytes(), type=Journal) if path.is_file() else empty

    def event(self, kind: str, state: str | None, frames: int | None) -> None:
        with self.lock:
            event = Event(now(), kind, state, frames)
            self.journal.events = [*self.journal.events, event][-SHOWN_EVENTS:]
            if self._progress(kind, state):
                self.journal.progressed = event.at
            self._save()

    def _progress(self, kind: str, state: str | None) -> bool:
        if kind != FOUND:
            return kind == IMPROVED
        if state is None or state in self.journal.found:
            return False
        self.journal.found = [*self.journal.found, state]
        return True

    def closer(self, state: str, remaining: int) -> None:
        with self.lock:
            closest = self.journal.closest
            if state in closest and remaining >= closest[state]:
                return
            was = f" (was {closest[state]})" if state in closest else ""
            closest[state] = remaining
            self._milestone("closer", f"{spoken(state)}: {remaining} frames before the door{was}")
            self._save()

    def progressed(self) -> datetime | None:
        with self.lock:
            progressed = self.journal.progressed
        return None if progressed is None else datetime.fromisoformat(progressed)

    def track(self, splits: list[Split], seconds: float, levels: Collection[str]) -> dict[str, float]:
        with self.lock:
            for split in splits:
                self._split(split, split.level in levels)
            history = self.journal.history
            if not history or history[-1].seconds != seconds:
                self.journal.history = [*history, Point(now(), seconds)][-SHOWN_HISTORY:]
            self._save()
            return dict(self.journal.firsts)

    def fields(self) -> dict[str, Any]:
        with self.lock:
            journal = self.journal
            return msgspec.to_builtins(
                {
                    "events": journal.events,
                    "history": journal.history,
                    "milestones": journal.milestones,
                    "progressed": journal.progressed,
                    "closest": journal.closest,
                }
            )

    def _split(self, split: Split, level: bool) -> None:
        firsts, bests = self.journal.firsts, self.journal.bests
        if split.level not in firsts:
            firsts[split.level] = bests[split.level] = split.level_seconds
            if level:
                self._milestone("level", f"{split.level} beaten in {clock(split.level_seconds)}")
            return
        if level and split.level_seconds < bests[split.level]:
            saved = bests[split.level] - split.level_seconds
            bests[split.level] = split.level_seconds
            self._milestone("best", f"{split.level} in {clock(split.level_seconds)}, {saved:.1f} s faster")

    def _milestone(self, kind: str, text: str) -> None:
        self.journal.milestones = [*self.journal.milestones, Milestone(now(), kind, text)][-SHOWN_MILESTONES:]

    def _save(self) -> None:
        atomic_write(self.path, msgspec.json.encode(self.journal))
