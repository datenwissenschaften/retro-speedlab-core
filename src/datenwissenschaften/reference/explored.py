import random
from collections import defaultdict
from pathlib import Path

import msgspec

from datenwissenschaften.route.route_map import Cell
from datenwissenschaften.speedrun.archive_store import ARCHIVE_SUFFIX, StoredArchive, hashable
from datenwissenschaften.speedrun.runner import Start, route_cell

STARTS_PER_CELL = 2


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
    for path in sorted(directory.rglob(f"*{ARCHIVE_SUFFIX}")):
        stored = msgspec.msgpack.decode(path.read_bytes(), type=StoredArchive)
        for cell in stored.cells:
            place = route_cell(hashable(cell.key))
            if place is None:
                continue
            seen[place] += 1
            keep(kept[place], seen[place], Start(stored.state, cell.emulator_state), rng)
    return [start for _, starts in sorted(kept.items()) for start in starts]
