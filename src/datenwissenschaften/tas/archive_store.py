import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import msgspec
import numpy as np

from datenwissenschaften.tas.novelty import BYTE_VALUES, RamNovelty, novelty_of
from datenwissenschaften.tas.runner import CellKey, Start
from datenwissenschaften.tas.search import Cell, Trail

ARCHIVE_SUFFIX = ".msgpack"


class StoredCell(msgspec.Struct, array_like=True):
    key: list[Any]
    emulator_state: bytes
    pressed: bytes | None
    buttons: int
    frames: int
    visits: int
    remaining: int | None
    score: float
    novelty: int


class StoredArchive(msgspec.Struct, array_like=True):
    state: str
    seen: bytes
    addresses: int
    cells: list[StoredCell]


def hashable(value: Any) -> Any:
    return tuple(hashable(item) for item in cast(list[Any], value)) if isinstance(value, list) else value


def stored_cell(key: CellKey, cell: Cell) -> StoredCell:
    pressed = None if cell.trail is None else cell.trail.frames()
    return StoredCell(
        list(key),
        cell.start.emulator_state,
        None if pressed is None else pressed.tobytes(),
        0 if pressed is None else pressed.shape[1],
        cell.frames,
        cell.visits,
        cell.remaining,
        cell.score,
        cell.novelty,
    )


def loaded_cell(state: str, stored: StoredCell) -> Cell:
    trail = None
    if stored.pressed is not None:
        trail = Trail(None, np.frombuffer(stored.pressed, dtype=np.int8).reshape(-1, stored.buttons).copy())
    start = Start(state, stored.emulator_state)
    return Cell(start, trail, stored.frames, stored.visits, stored.remaining, stored.score, stored.novelty)


def save_archive(path: Path, state: str, cells: dict[CellKey, Cell], novelty: RamNovelty) -> None:
    seen = np.packbits(novelty.seen).tobytes()
    stored = StoredArchive(state, seen, len(novelty.seen), [stored_cell(key, cell) for key, cell in cells.items()])
    written = path.with_name(f"{path.name}.{os.getpid()}")
    written.write_bytes(msgspec.msgpack.encode(stored))
    written.replace(path)


def load_archive(path: Path) -> tuple[dict[CellKey, Cell], RamNovelty]:
    stored = msgspec.msgpack.decode(path.read_bytes(), type=StoredArchive)
    bits = np.unpackbits(np.frombuffer(stored.seen, dtype=np.uint8), count=stored.addresses * BYTE_VALUES)
    novelty = novelty_of(bits.astype(bool).reshape(stored.addresses, BYTE_VALUES))
    return {hashable(cell.key): loaded_cell(stored.state, cell) for cell in stored.cells}, novelty


def modified(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def newest(paths: Iterable[Path]) -> list[Path]:
    times = {path: time for path in paths if (time := modified(path)) is not None}
    return sorted(times, key=lambda path: times[path], reverse=True)
