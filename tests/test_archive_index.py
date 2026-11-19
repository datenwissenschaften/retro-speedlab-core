import random

import numpy as np

from datenwissenschaften.speedrun.archive_index import ArchiveIndex
from datenwissenschaften.speedrun.runner import CellKey, Start
from datenwissenschaften.speedrun.search import EXPLOIT, LEADING_SHARE, Cell, Trail

CAPACITY = 60
OPERATIONS = 4000
KEYS = 150
SCORES = 12
START = Start("Walk", b"")
TRAIL = Trail(None, np.zeros((1, 1), dtype=np.int8))


def scanned_choice(cells: list[Cell], rng: random.Random) -> Cell:
    if rng.random() < EXPLOIT:
        cells = sorted(cells, key=lambda cell: cell.score)[-max(1, len(cells) // LEADING_SHARE) :]
    cell = rng.choices(cells, weights=[cell.weight for cell in cells])[0]
    cell.visits += 1
    return cell


def scanned_eviction(archive: dict[CellKey, Cell]) -> None:
    places = [kept for kept, cell in archive.items() if cell.trail is not None]
    del archive[min(places, key=lambda kept: (archive[kept].score, -archive[kept].visits))]


def random_cell(rng: random.Random) -> Cell:
    remaining = None if rng.random() < 0.3 else rng.randrange(40)
    return Cell(START, TRAIL, rng.randrange(100), rng.randrange(4), remaining, float(rng.randrange(SCORES)), 3)


def snapshot(archive: dict[CellKey, Cell]) -> list[tuple[CellKey, int, int, float]]:
    return [(key, cell.frames, cell.visits, cell.score) for key, cell in archive.items()]


def test_the_index_evicts_and_chooses_exactly_like_scanning_every_cell():
    seed = random.Random(5)
    root = Cell(START, None, 0, 0, None, 0.0, 0)
    scanned: dict[CellKey, Cell] = {("root",): root}
    indexed: dict[CellKey, Cell] = {("root",): Cell(START, None, 0, 0, None, 0.0, 0)}
    index = ArchiveIndex(indexed)
    first, second = random.Random(9), random.Random(9)
    for _ in range(OPERATIONS):
        roll = seed.random()
        if roll < 0.3:
            scanned_choice(list(scanned.values()), first)
            index.choose(second)
        elif roll < 0.32:
            for scanned_cell, indexed_cell in zip(scanned.values(), indexed.values(), strict=True):
                scanned_cell.remaining = indexed_cell.remaining = seed.randrange(40)
            index.reweigh()
        else:
            key, cell = (seed.randrange(KEYS),), random_cell(seed)
            if key not in scanned and len(scanned) >= CAPACITY:
                scanned_eviction(scanned)
                index.evict()
            scanned[key] = cell
            index.put(key, Cell(START, TRAIL, cell.frames, cell.visits, cell.remaining, cell.score, cell.novelty))
        assert snapshot(scanned) == snapshot(indexed)
    assert list(index.weights) == list(indexed)
