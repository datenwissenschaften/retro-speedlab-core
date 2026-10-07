import base64
import threading
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from datenwissenschaften.ui.replay_store import ReplayStore
from datenwissenschaften.ui.replay_video import encode_video

MAX_COMPLETED_EPISODES = 4
MAX_STATUSES_PER_REQUEST = 600


class LiveFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._store: ReplayStore | None = None
        self._start_generation()

    def keep_replays_in(self, directory: Path, curricula: frozenset[str]) -> None:
        with self._lock:
            self._store = ReplayStore(directory)
            self._replays = {name: replay for name, replay in self._replays.items() if name in curricula}
            self._replays.update(self._store.load(curricula))

    def clear(self) -> None:
        with self._lock:
            self._start_generation()
            if self._store is not None:
                self._store.clear()

    def _start_generation(self) -> None:
        self._generation = uuid4().hex
        self._recording: list[dict[str, Any]] = []
        self._episodes: deque[dict[str, Any]] = deque(maxlen=MAX_COMPLETED_EPISODES)
        self._replays: dict[str, dict[str, Any]] = {}
        self._summary: dict[str, Any] = {}

    def record(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._recording.append({"image": jpeg, "status": status})

    def last_status(self) -> dict[str, Any]:
        with self._lock:
            if not self._recording:
                raise RuntimeError("No frame has been recorded for this episode.")
            return self._recording[-1]["status"]

    def last_image(self) -> str:
        with self._lock:
            if not self._recording:
                raise RuntimeError("No frame has been recorded for this episode.")
            return base64.b64encode(self._recording[-1]["image"]).decode("ascii")

    def finish_episode(
        self, episode_id: int, frame_rate: float, result: dict[str, Any], summary: dict[str, Any]
    ) -> None:
        with self._lock:
            frames, self._recording = self._recording, []
            self._summary = summary
        if not frames:
            return
        episode = {
            "id": episode_id,
            "frame_rate": frame_rate,
            "result": result,
            "statuses": [frame["status"] for frame in frames],
            "video": encode_video([frame["image"] for frame in frames], frame_rate),
        }
        with self._lock:
            self._episodes.append(episode)
            self._keep_best(episode)

    def _keep_best(self, episode: dict[str, Any]) -> None:
        curriculum = episode["result"]["curriculum"]
        if curriculum not in self._replays or _rank(episode) > _rank(self._replays[curriculum]):
            self._replays[curriculum] = episode
            if self._store is not None:
                self._store.save(curriculum, episode)

    def latest_episode(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "episode": _overview(self._episodes[-1]) if self._episodes else None,
                "replays": [_overview(episode) for episode in self._replays.values()],
                "summary": dict(self._summary),
            }

    def episode_statuses(self, generation: str, episode_id: int, start: int) -> list[dict[str, Any]]:
        statuses = self._episode(generation, episode_id)["statuses"]
        return [dict(status) for status in statuses[start : start + MAX_STATUSES_PER_REQUEST]]

    def episode_video(self, generation: str, episode_id: int) -> bytes:
        return self._episode(generation, episode_id)["video"]

    def _episode(self, generation: str, episode_id: int) -> dict[str, Any]:
        with self._lock:
            if generation != self._generation:
                raise KeyError(generation)
            for episode in (*self._episodes, *self._replays.values()):
                if episode["id"] == episode_id:
                    return episode
        raise KeyError(episode_id)


def _rank(episode: dict[str, Any]) -> tuple[bool, float]:
    result = episode["result"]
    return (True, -len(episode["statuses"])) if result["succeeded"] else (False, result["score"])


def _overview(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": episode["id"],
        "frame_rate": episode["frame_rate"],
        "frame_count": len(episode["statuses"]),
        "result": episode["result"],
    }


live_feed = LiveFeed()
