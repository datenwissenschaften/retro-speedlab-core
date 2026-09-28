import re
from collections import Counter
from datetime import date
from typing import Any

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.ui.telemetry import publish_metadata

STORY_KEY = "story"
DANGER_WINDOW = 100
DANGER_CELL = 32
DANGER_SPOTS = 3
KEPT_IMAGES = 2 * DANGER_SPOTS
WORD_BOUNDARY = re.compile(r"(?<=[a-z])(?=[A-Z])|_")


def story_key(identity: str) -> str:
    return f"{STORY_KEY}:{identity}"


def label(name: str) -> str:
    return WORD_BOUNDARY.sub(" ", name).strip().title()


def empty_story() -> dict[str, Any]:
    return {
        "reached": {},
        "day": date.today().isoformat(),
        "failures_today": {},
        "failures": [],
        "images": {},
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

    def finish(self, phase: str, succeeded: bool, location: tuple[int, int] | None, image: str) -> int:
        if succeeded:
            return 0
        self._remember_danger(phase, location, image)
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
            "danger": [
                {
                    "phase": label(self.data["images"][spot]["phase"]),
                    "count": count,
                    "image": self.data["images"][spot]["image"],
                }
                for spot, count in self._ranked_spots()[:DANGER_SPOTS]
            ],
        }

    def _phase_view(self, phase: str) -> dict[str, Any]:
        return {
            "name": phase,
            "label": label(phase),
            "reached": self.has_reached(phase),
            "first_attempt": self.data["reached"][phase] if self.has_reached(phase) else None,
        }

    def _remember_danger(self, phase: str, location: tuple[int, int] | None, image: str) -> None:
        spot = phase if location is None else f"{phase}@{location[0] // DANGER_CELL},{location[1] // DANGER_CELL}"
        self.data["failures"].append(spot)
        del self.data["failures"][:-DANGER_WINDOW]
        self.data["images"][spot] = {"phase": phase, "image": image}
        kept = {kept_spot for kept_spot, _ in self._ranked_spots()[:KEPT_IMAGES]} | {spot}
        self.data["images"] = {key: value for key, value in self.data["images"].items() if key in kept}

    def _ranked_spots(self) -> list[tuple[str, int]]:
        counts = Counter(spot for spot in self.data["failures"] if spot in self.data["images"])
        return counts.most_common()

    @staticmethod
    def _increment(counts: dict[str, int], key: str) -> int:
        counts[key] = (counts[key] if key in counts else 0) + 1
        return counts[key]
