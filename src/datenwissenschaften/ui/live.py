import base64
import threading
from collections import deque
from typing import Any

MAX_COMPLETED_EPISODES = 2
MAX_FRAMES_PER_REQUEST = 120


class LiveFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recording: list[dict[str, Any]] = []
        self._episodes: deque[dict[str, Any]] = deque(maxlen=MAX_COMPLETED_EPISODES)
        self._summary: dict[str, Any] = {}

    def record(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._recording.append({"image": base64.b64encode(jpeg).decode("ascii"), "status": status})

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
                self._episodes.append({"id": episode_id, "frame_rate": frame_rate, "result": result, "frames": frames})
            self._summary = summary

    def latest_episode(self) -> dict[str, Any]:
        with self._lock:
            if not self._episodes:
                return {"episode": None, "summary": dict(self._summary)}
            latest = self._episodes[-1]
            episode = {
                "id": latest["id"],
                "frame_rate": latest["frame_rate"],
                "frame_count": len(latest["frames"]),
                "result": latest["result"],
            }
            return {"episode": episode, "summary": dict(self._summary)}

    def episode_frames(self, episode_id: int, start: int) -> list[dict[str, Any]]:
        with self._lock:
            for episode in self._episodes:
                if episode["id"] == episode_id:
                    return episode["frames"][start : start + MAX_FRAMES_PER_REQUEST]
        raise KeyError(episode_id)


live_feed = LiveFeed()
