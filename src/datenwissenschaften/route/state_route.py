import fcntl
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.route.position import Position

CELL_SIZE = 8
RELOAD_SECONDS = 10.0
RESPAWN_DISTANCE = 64

Cell = tuple[int, int, int, int, bool]
Step = tuple[Position, int]


@dataclass(slots=True, frozen=True)
class Move:
    action: int
    remaining: int


def cell_of(position: Position) -> Cell:
    size = CELL_SIZE
    return position.area, position.x // size, position.y // size, position.height // size, position.airborne


def teleported(before: Position, after: Position) -> bool:
    if before.area != after.area:
        return True
    distance = abs(after.x - before.x) + abs(after.y - before.y) + abs(after.height - before.height)
    return distance > RESPAWN_DISTANCE


class StateRoute:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.moves: dict[Cell, Move] = {}
        self.centers: dict[int, np.ndarray] = {}
        self.version: int | None = None
        self.checked_at = -math.inf

    def move(self, position: Position) -> Move | None:
        self._refresh()
        cell = cell_of(position)
        return self.moves[cell] if cell in self.moves else None

    def nearest(self, position: Position) -> tuple[int, int] | None:
        self._refresh()
        if position.area not in self.centers:
            return None
        centers = self.centers[position.area]
        here = np.array([position.x, position.y, position.height])
        x, y, _ = centers[np.abs(centers - here).sum(axis=1).argmin()]
        return int(x), int(y)

    def record(self, trail: list[Step]) -> None:
        with self._lock():
            self._load()
            for index, (position, action) in enumerate(trail):
                cell, remaining = cell_of(position), len(trail) - index
                if cell not in self.moves or remaining <= self.moves[cell].remaining:
                    self.moves[cell] = Move(action, remaining)
            self._write()
            self._index()

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self.checked_at < RELOAD_SECONDS:
            return
        self.checked_at = now
        if self._stamp() != self.version:
            self._load()
            self._index()

    def _load(self) -> None:
        self.version = self._stamp()
        if self.version is None:
            self.moves = {}
            return
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        self.moves = {(a, x, y, h, bool(air)): Move(action, left) for a, x, y, h, air, action, left in rows}

    def _write(self) -> None:
        rows = [[*cell, move.action, move.remaining] for cell, move in self.moves.items()]
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)
        self.version = self._stamp()

    def _index(self) -> None:
        half = CELL_SIZE // 2
        areas: dict[int, list[tuple[int, int, int]]] = {}
        for area, x, y, height, _ in self.moves:
            areas.setdefault(area, []).append((x * CELL_SIZE + half, y * CELL_SIZE + half, height * CELL_SIZE + half))
        self.centers = {area: np.array(cells) for area, cells in areas.items()}

    def _stamp(self) -> int | None:
        return self.path.stat().st_mtime_ns if self.path.is_file() else None

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_name(f".{self.path.name}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
