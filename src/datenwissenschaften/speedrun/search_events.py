from typing import Any

from datenwissenschaften.events import ATTEMPTS, RECORDS, STEPS, Events, now
from datenwissenschaften.speedrun.attempt import Attempt, Origin
from datenwissenschaften.speedrun.movie import Segment, frames
from datenwissenschaften.speedrun.search import Searched

CROSSING = "crossing"
CHECKPOINT = "checkpoint"
NOTHING = "nothing"


def outcome(searched: Searched) -> str:
    if searched.crossings:
        return CROSSING
    return NOTHING if searched.checkpoint is None else CHECKPOINT


class SearchEvents:
    def __init__(self, events: Events, game: str, version: str, frame_rate: float) -> None:
        self.events = events
        self.game = game
        self.version = version
        self.frame_rate = frame_rate

    def attempt(self, attempt: Attempt, seconds: float) -> None:
        searched = attempt.searched
        fastest = min(searched.crossings, key=lambda crossing: crossing.frames, default=None)
        ended = fastest if fastest is not None else searched.checkpoint
        payload = {
            "state": attempt.state,
            "seed": attempt.seed,
            "role": attempt.origin.role,
            "seconds": seconds,
            "worker_seconds": attempt.origin.worker_seconds,
            "outcome": outcome(searched),
            "crossings": len(searched.crossings),
            "frames": None if ended is None else ended.frames,
            "reached": None if ended is None else ended.end.state,
        }
        self.publish(ATTEMPTS, payload)

    def step(self, kind: str, state: str | None, frames: int | None, saved: int | None, origin: Origin | None) -> None:
        payload = {
            "kind": kind,
            "state": state,
            "frames": frames,
            "saved": saved,
            "role": None if origin is None else origin.role,
            "worker_seconds": None if origin is None else origin.worker_seconds,
        }
        self.publish(STEPS, payload)

    def record(self, segments: list[Segment], previous: list[Segment]) -> None:
        payload = {
            "version": self.version,
            "frames": frames(segments),
            "seconds": frames(segments) / self.frame_rate,
            "pieces": len(segments),
            "previous_frames": frames(previous) if previous else None,
        }
        self.publish(RECORDS, payload)

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        self.events.publish(topic, self.game, {"game": self.game, **payload, "at": now()})
