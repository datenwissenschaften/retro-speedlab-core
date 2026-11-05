import random
from collections import defaultdict
from pathlib import Path

import msgspec

from datenwissenschaften.route.route_map import Cell
from datenwissenschaften.tas.archive_store import ARCHIVE_SUFFIX, StoredArchive, hashable
from datenwissenschaften.tas.runner import Start, route_cell

STARTS_PER_CELL = 2


def archived_starts(directory: Path) -> dict[Cell, list[Start]]:
    found: defaultdict[Cell, list[Start]] = defaultdict(list)
    for path in sorted(directory.rglob(f"*{ARCHIVE_SUFFIX}")):
        stored = msgspec.msgpack.decode(path.read_bytes(), type=StoredArchive)
        for cell in stored.cells:
            place = route_cell(hashable(cell.key))
            if place is not None:
                found[place].append(Start(stored.state, cell.emulator_state))
    return found


def explored_starts(directory: Path, rng: random.Random) -> list[Start]:
    found = archived_starts(directory)
    return [
        start for _, starts in sorted(found.items()) for start in rng.sample(starts, min(len(starts), STARTS_PER_CELL))
    ]
