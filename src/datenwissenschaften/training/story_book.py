import re
from datetime import date
from typing import Any

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.ui.telemetry import publish_metadata

STORY_KEY = "story"
SKILL_WINDOW = 20
MAP_CELL = 16
WORD_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|_")


def story_key(identity: str) -> str:
    return f"{STORY_KEY}:{identity}"


def label(name: str) -> str:
    return WORD_BOUNDARY.sub(" ", name).strip().title()


def empty_story() -> dict[str, Any]:
    return {
        "reached": {},
        "outcomes": {},
        "day": date.today().isoformat(),
        "failures_today": {},
        "visits": {},
        "ends": {},
    }


class StoryBook:
    def __init__(self, database: JsonDatabase, identity: str, phases: tuple[str, ...]) -> None:
        self.database = database
        self.key = story_key(identity)
        self.phases = phases
        self.data = database.get(self.key) if database.contains(self.key) else empty_story()
        self.publish()

    def has_reached(self, phase: str) -> bool:
        return phase in self.data["reached"]

    def reach(self, phase: str, attempt: int) -> None:
        self.data["reached"][phase] = attempt

    def visit(self, location: tuple[int, int]) -> None:
        self._count("visits", location)

    def finish(self, phase: str, succeeded: bool, location: tuple[int, int] | None) -> int:
        outcomes = self.data["outcomes"].setdefault(phase, [])
        outcomes.append(succeeded)
        del outcomes[:-SKILL_WINDOW]
        if succeeded:
            return 0
        if location is not None:
            self._count("ends", location)
        today = date.today().isoformat()
        if self.data["day"] != today:
            self.data["day"], self.data["failures_today"] = today, {}
        return self._increment(self.data["failures_today"], phase)

    def save(self) -> None:
        self.database.set(self.key, self.data)
        self.publish()

    def publish(self) -> None:
        publish_metadata("story", self.view(), replace=True)

    def view(self) -> dict[str, Any]:
        return {
            "phases": [self._phase_view(phase) for phase in self.phases],
            "map": {"cell": MAP_CELL, "visits": self._cells("visits"), "ends": self._cells("ends")},
        }

    def _phase_view(self, phase: str) -> dict[str, Any]:
        outcomes = self.data["outcomes"][phase] if phase in self.data["outcomes"] else []
        return {
            "name": phase,
            "label": label(phase),
            "reached": self.has_reached(phase),
            "first_attempt": self.data["reached"][phase] if self.has_reached(phase) else None,
            "skill": sum(outcomes) / len(outcomes) if outcomes else None,
        }

    def _count(self, kind: str, location: tuple[int, int]) -> None:
        self._increment(self.data[kind], f"{location[0] // MAP_CELL},{location[1] // MAP_CELL}")

    @staticmethod
    def _increment(counts: dict[str, int], key: str) -> int:
        counts[key] = (counts[key] if key in counts else 0) + 1
        return counts[key]

    def _cells(self, kind: str) -> list[list[int]]:
        return [[*map(int, cell.split(",")), count] for cell, count in self.data[kind].items()]
