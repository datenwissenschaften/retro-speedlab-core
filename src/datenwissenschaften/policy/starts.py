import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.speedrun.archive_store import ARCHIVE_SUFFIX, load_archive, newest
from datenwissenschaften.speedrun.archives import archive_key
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Cell

FRONTIER_SHARE = 0.75
FRONTIER_PLACES = 20


@dataclass(slots=True, frozen=True)
class Opening:
    start: Start
    prefix: np.ndarray


def frontier(cell: Cell) -> tuple[bool, int, float, int]:
    return cell.remaining is None, cell.remaining or 0, -cell.score, -cell.frames


def openings(archives: Path, start: Start, buttons: int) -> list[Opening]:
    root = Opening(start, np.zeros((0, buttons), dtype=np.int8))
    saved = newest((archives / archive_key(start, UNBOUNDED)).glob(f"*{ARCHIVE_SUFFIX}"))
    if not saved:
        return [root]
    cells, _ = load_archive(saved[0])
    placed = sorted((cell for cell in cells.values() if cell.trail is not None), key=frontier)
    return [root, *(Opening(cell.start, cell.trail.frames()) for cell in placed if cell.trail is not None)]


def chosen(found: list[Opening], rng: random.Random) -> Opening:
    if len(found) > 1 and rng.random() < FRONTIER_SHARE:
        return rng.choice(found[1 : FRONTIER_PLACES + 1])
    return rng.choice(found)
