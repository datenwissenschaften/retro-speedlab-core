import shutil
from pathlib import Path
from typing import Any

import msgspec
from loguru import logger

from datenwissenschaften.ui.media import episode_key

REPLAY_SUFFIX = ".replay"
PARTIAL_SUFFIX = ".partial"


class ReplayStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def load(self, curricula: frozenset[str]) -> dict[str, list[dict[str, Any]]]:
        replays = {}
        for path in sorted(self.directory.iterdir()):
            if path.name not in curricula:
                logger.info(f"Deleting the replays in {path.name}: the curriculum no longer has it")
                _remove(path)
                continue
            replays[path.name] = sorted(self._load_state(path), key=lambda episode: episode["id"])
        return replays

    def _load_state(self, directory: Path) -> list[dict[str, Any]]:
        episodes = []
        for path in directory.glob(f"*{REPLAY_SUFFIX}"):
            episode = msgspec.msgpack.decode(path.read_bytes())
            if not episode["result"]["succeeded"]:
                logger.info(f"Deleting the replay {path.name} of {directory.name}: it never reached its goal")
                path.unlink()
                continue
            episode["key"] = episode_key(episode)
            episodes.append(episode)
        return episodes

    def save(self, episode: dict[str, Any]) -> None:
        path = self._path(episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(PARTIAL_SUFFIX)
        partial.write_bytes(msgspec.msgpack.encode(episode))
        partial.replace(path)

    def delete(self, episode: dict[str, Any]) -> None:
        self._path(episode).unlink(missing_ok=True)

    def drop(self, curriculum: str) -> None:
        _remove(self.directory / curriculum)

    def clear(self) -> None:
        for path in self.directory.iterdir():
            _remove(path)

    def _path(self, episode: dict[str, Any]) -> Path:
        return self.directory / episode["result"]["curriculum"] / f"{episode['key']}{REPLAY_SUFFIX}"


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
        return
    path.unlink(missing_ok=True)
