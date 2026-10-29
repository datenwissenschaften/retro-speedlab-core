import threading
from collections import deque
from typing import Any
from uuid import uuid4

from datenwissenschaften.ui.media import episode_key, episode_media, episode_media_names
from datenwissenschaften.ui.replay_video import encode_video

MAX_COMPLETED_EPISODES = 4


class LiveFeed:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = uuid4().hex
        self._recording: list[dict[str, Any]] = []
        self._episodes: deque[dict[str, Any]] = deque(maxlen=MAX_COMPLETED_EPISODES)

    def record(self, jpeg: bytes, status: dict[str, Any]) -> None:
        with self._lock:
            self._recording.append({"image": jpeg, "status": status})

    def finish_episode(self, episode_id: int, frame_rate: float, result: dict[str, Any]) -> None:
        with self._lock:
            frames, self._recording = self._recording, []
        if not frames:
            return
        episode = {
            "id": episode_id,
            "frame_rate": frame_rate,
            "result": result,
            "statuses": [frame["status"] for frame in frames],
            "video": encode_video([frame["image"] for frame in frames], frame_rate),
        }
        episode["key"] = episode_key(episode)
        with self._lock:
            self._episodes.append(episode)

    def latest_episode(self) -> dict[str, Any]:
        with self._lock:
            latest = _overview(self._episodes[-1]) if self._episodes else None
            return {"generation": self._generation, "episode": latest}

    def media_names(self) -> set[str]:
        with self._lock:
            return {name for episode in self._episodes for name in episode_media_names(episode)}

    def media(self, name: str) -> bytes:
        with self._lock:
            episodes = [episode for episode in self._episodes if name in episode_media_names(episode)]
        if not episodes:
            raise KeyError(name)
        return episode_media(episodes[0], name)


def _overview(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": episode["id"],
        "frame_rate": episode["frame_rate"],
        "frame_count": len(episode["statuses"]),
        "result": episode["result"],
        "key": episode["key"],
    }


live_feed = LiveFeed()
