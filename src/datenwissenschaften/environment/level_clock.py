import json
import os
from pathlib import Path

from datenwissenschaften.ui.telemetry import publish_metadata


class LevelClock:
    def __init__(self, path: Path, frame_rate: float) -> None:
        self.path = path
        self.frame_rate = frame_rate

    def times(self) -> dict[str, dict[str, float | int]]:
        if not self.path.is_file():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def record(self, level: str, frames: int) -> None:
        seconds = frames / self.frame_rate
        times = self.times()
        previous = times[level] if level in times else {"best_seconds": seconds, "runs": 0}
        times[level] = {
            "best_seconds": min(float(previous["best_seconds"]), seconds),
            "last_seconds": seconds,
            "runs": int(previous["runs"]) + 1,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(times), encoding="utf-8")
        temporary.replace(self.path)
        self.publish()

    def publish(self) -> None:
        publish_metadata("level_times", self.times(), replace=True)
