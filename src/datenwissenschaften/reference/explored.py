import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import msgspec

from datenwissenschaften.route.route_map import Cell
from datenwissenschaften.speedrun.archive_store import ARCHIVE_SUFFIX, StoredArchive, hashable
from datenwissenschaften.speedrun.runner import CellKey, Start, route_cell

STARTS_PER_CELL = 2


def placed_cells(directory: Path) -> Iterator[tuple[CellKey, Cell, Start]]:
    for path in sorted(directory.rglob(f"*{ARCHIVE_SUFFIX}")):
        stored = msgspec.msgpack.decode(path.read_bytes(), type=StoredArchive)
        for cell in stored.cells:
            key = hashable(cell.key)
            place = route_cell(key)
            if place is not None:
                yield key, place, Start(stored.state, cell.emulator_state)


def keep(kept: list[Start], seen: int, start: Start, rng: random.Random) -> None:
    if len(kept) < STARTS_PER_CELL:
        kept.append(start)
        return
    slot = rng.randrange(seen)
    if slot < STARTS_PER_CELL:
        kept[slot] = start


def explored_starts(directory: Path, rng: random.Random) -> list[Start]:
    kept: defaultdict[Cell, list[Start]] = defaultdict(list)
    seen: defaultdict[Cell, int] = defaultdict(int)
    for _, place, start in placed_cells(directory):
        seen[place] += 1
        keep(kept[place], seen[place], start, rng)
    return [start for _, starts in sorted(kept.items()) for start in starts]


def archived_starts(directory: Path) -> list[Start]:
    found: dict[tuple[str, CellKey], Start] = {}
    for key, _, start in placed_cells(directory):
        found.setdefault((start.state, key), start)
    return list(found.values())
