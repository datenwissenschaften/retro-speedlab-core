import hashlib
from dataclasses import dataclass

from cachetools import LRUCache

from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.tas.search import Cell

KEPT_ARCHIVES = 12
KEPT_CELLS = 1000

ArchiveKey = tuple[str, bytes, int]


@dataclass(slots=True)
class Archive:
    cells: dict[CellKey, Cell]
    novelty: RamNovelty


def fresh_archive(runner: SegmentRunner, start: Start) -> Archive:
    runner.restore(start)
    root = Cell(start, None, 0, 0, runner.remaining(), 0.0, 0)
    return Archive({runner.key(0): root}, RamNovelty(runner.memory()))


def pruned(archive: Archive) -> Archive:
    ranked = sorted(archive.cells.items(), key=lambda item: (item[1].score, -item[1].visits), reverse=True)
    return Archive(dict(ranked[:KEPT_CELLS]), archive.novelty)


def archive_key(start: Start, bound: int) -> ArchiveKey:
    return start.state, hashlib.sha256(start.emulator_state).digest(), bound


class Archives:
    def __init__(self) -> None:
        self.kept: LRUCache[ArchiveKey, Archive] = LRUCache(maxsize=KEPT_ARCHIVES)

    def resumed(self, runner: SegmentRunner, start: Start, bound: int) -> Archive:
        key = archive_key(start, bound)
        if key in self.kept:
            return self.kept[key]
        return fresh_archive(runner, start)

    def keep(self, start: Start, bound: int, archive: Archive) -> None:
        self.kept[archive_key(start, bound)] = pruned(archive)
