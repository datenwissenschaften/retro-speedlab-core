import base64
import threading
from collections import deque
from typing import Any
from uuid import uuid4

MAX_COMPLETED_EPISODES = 2
MAX_REPLAYS = 3
MAX_FRAMES_PER_REQUEST = 120


class LiveFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_generation()

    def clear(self) -> None:
        with self._lock:
            self._start_generation()

    def _start_generation(self) -> None:
        self._generation = uuid4().hex
        self._recording: list[dict[str, Any]] = []
        self._episodes: deque[dict[str, Any]] = deque(maxlen=MAX_COMPLETED_EPISODES)
        self._replays: deque[dict[str, Any]] = deque(maxlen=MAX_REPLAYS)
        self._summary: dict[str, Any] = {}

    def record(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._recording.append({"image": base64.b64encode(jpeg).decode("ascii"), "status": status})

    def last_status(self) -> dict[str, Any]:
        with self._lock:
            if not self._recording:
                raise RuntimeError("No frame has been recorded for this episode.")
            return self._recording[-1]["status"]

    def last_image(self) -> str:
        with self._lock:
            if not self._recording:
                raise RuntimeError("No frame has been recorded for this episode.")
            return self._recording[-1]["image"]

    def add_events(self, events: list[dict[str, str]]) -> None:
        with self._lock:
            if self._recording:
                status = self._recording[-1]["status"]
                status["events"] = [*status["events"], *events]

    def finish_episode(
        self, episode_id: int, frame_rate: float, result: dict[str, Any], summary: dict[str, Any]
    ) -> None:
        with self._lock:
            frames, self._recording = self._recording, []
            if frames:
                episode = {"id": episode_id, "frame_rate": frame_rate, "result": result, "frames": frames}
                self._episodes.append(episode)
                if result["full_run"] or result["succeeded"]:
                    self._replays.append(episode)
            self._summary = summary

    def latest_episode(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "episode": _overview(self._episodes[-1]) if self._episodes else None,
                "replays": [_overview(episode) for episode in self._replays],
                "summary": dict(self._summary),
            }

    def episode_frames(self, generation: str, episode_id: int, start: int) -> list[dict[str, Any]]:
        with self._lock:
            if generation != self._generation:
                raise KeyError(generation)
            for episode in (*self._episodes, *self._replays):
                if episode["id"] == episode_id:
                    frames = episode["frames"][start : start + MAX_FRAMES_PER_REQUEST]
                    return [{"image": frame["image"], "status": dict(frame["status"])} for frame in frames]
        raise KeyError(episode_id)


def _overview(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": episode["id"],
        "frame_rate": episode["frame_rate"],
        "frame_count": len(episode["frames"]),
        "result": episode["result"],
    }


live_feed = LiveFeed()
