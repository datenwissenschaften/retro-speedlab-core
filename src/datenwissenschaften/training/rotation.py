import time
from collections.abc import Callable
from typing import Any

from datenwissenschaften.persistence import JsonDatabase

ROTATION_KEY = "rotation"
SECONDS_PER_MINUTE = 60
BEATEN_FULL_RUN_WINS = 8


class Rotation:
    def __init__(
        self,
        database: JsonDatabase,
        identity: str,
        savestates: tuple[str, ...],
        minutes: int,
        full_run_wins: Callable[[str], int],
    ) -> None:
        self.database = database
        self.key = f"{ROTATION_KEY}:{identity}"
        self.savestates = savestates
        self.period = minutes * SECONDS_PER_MINUTE
        self.full_run_wins = full_run_wins

    def next(self) -> tuple[str, float, bool]:
        unbeaten = [savestate for savestate in self.savestates if self.full_run_wins(savestate) < BEATEN_FULL_RUN_WINS]
        if unbeaten:
            return unbeaten[0], self.period, False
        savestate, seconds = self._speedrun_turn()
        return savestate, seconds, True

    def _speedrun_turn(self) -> tuple[str, float]:
        now = time.time()
        stored = self.database.get(self.key) if self.database.contains(self.key) else None
        if stored is not None and stored["savestate"] in self.savestates and stored["ends_at"] > now:
            return stored["savestate"], stored["ends_at"] - now
        savestate = self._following(stored)
        self.database.set(self.key, {"savestate": savestate, "ends_at": now + self.period})
        return savestate, self.period

    def _following(self, stored: dict[str, Any] | None) -> str:
        if stored is None or stored["savestate"] not in self.savestates:
            return self.savestates[0]
        return self.savestates[(self.savestates.index(stored["savestate"]) + 1) % len(self.savestates)]
