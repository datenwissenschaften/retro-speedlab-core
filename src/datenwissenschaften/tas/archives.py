import hashlib
from dataclasses import dataclass
from pathlib import Path

from cachetools import LRUCache

from datenwissenschaften.tas.archive_store import ARCHIVE_SUFFIX, load_archive, prune_archives, save_archive
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.tas.search import Cell

KEPT_ARCHIVES = 12
KEPT_CELLS = 1000

ArchiveKey = str


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
    return hashlib.sha256(b"|".join([start.state.encode(), start.emulator_state, str(bound).encode()])).hexdigest()


class Archives:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.kept: LRUCache[ArchiveKey, Archive] = LRUCache(maxsize=KEPT_ARCHIVES)
        directory.mkdir(parents=True, exist_ok=True)

    def path(self, key: ArchiveKey) -> Path:
        return self.directory / f"{key}{ARCHIVE_SUFFIX}"

    def resumed(self, runner: SegmentRunner, start: Start, bound: int) -> Archive:
        key = archive_key(start, bound)
        if key in self.kept:
            return self.kept[key]
        if self.path(key).is_file():
            return Archive(*load_archive(self.path(key)))
        return fresh_archive(runner, start)

    def keep(self, start: Start, bound: int, archive: Archive) -> None:
        key = archive_key(start, bound)
        kept = pruned(archive)
        self.kept[key] = kept
        save_archive(self.path(key), start.state, kept.cells, kept.novelty)
        prune_archives(self.directory, KEPT_ARCHIVES)
