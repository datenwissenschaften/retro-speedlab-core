from pathlib import Path
from typing import Any

import msgspec
from loguru import logger

REPLAY_SUFFIX = ".replay"
PARTIAL_SUFFIX = ".partial"


class ReplayStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, dict[str, Any]]:
        replays = {}
        for path in sorted(self.directory.glob(f"*{REPLAY_SUFFIX}")):
            episode = msgspec.msgpack.decode(path.read_bytes())
            if "video" not in episode:
                logger.warning(f"Deleting {path.name}: it stores frames instead of a video")
                path.unlink()
                continue
            replays[episode["result"]["curriculum"]] = episode
        return replays

    def save(self, curriculum: str, episode: dict[str, Any]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{curriculum}{REPLAY_SUFFIX}"
        partial = path.with_suffix(PARTIAL_SUFFIX)
        partial.write_bytes(msgspec.msgpack.encode(episode))
        partial.replace(path)

    def clear(self) -> None:
        for path in self.directory.glob(f"*{REPLAY_SUFFIX}"):
            path.unlink()
