from collections import defaultdict
from typing import Any

from datenwissenschaften.ui.replay_store import ReplayStore

MAX_REPLAYS_PER_STATE = 8


class ReplayRotation:
    def __init__(self) -> None:
        self._store: ReplayStore | None = None
        self._replays: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def persist_in(self, store: ReplayStore, curricula: frozenset[str]) -> None:
        self._store = store
        self._replays = defaultdict(list, {name: kept for name, kept in self._replays.items() if name in curricula})
        self._replays.update(store.load(curricula))

    def clear(self) -> None:
        self._replays = defaultdict(list)
        if self._store is not None:
            self._store.clear()

    def keep(self, episode: dict[str, Any]) -> None:
        if not episode["result"]["succeeded"]:
            return
        replays = self._replays[episode["result"]["curriculum"]]
        replays.append(episode)
        evicted = replays[:-MAX_REPLAYS_PER_STATE]
        del replays[:-MAX_REPLAYS_PER_STATE]
        if self._store is None:
            return
        self._store.save(episode)
        for oldest in evicted:
            self._store.delete(oldest)

    def drop(self, curricula: tuple[str, ...]) -> None:
        for curriculum in curricula:
            if curriculum in self._replays:
                del self._replays[curriculum]
            if self._store is not None:
                self._store.drop(curriculum)

    def episodes(self) -> list[dict[str, Any]]:
        return [episode for replays in self._replays.values() for episode in replays]
