import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cachetools import LRUCache

from datenwissenschaften.tas.archive_store import ARCHIVE_SUFFIX, load_archive, newest, save_archive
from datenwissenschaften.tas.novelty import RamNovelty, novelty_of
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.tas.search import Cell

KEPT_ARCHIVES = 12
KEPT_CELLS = 1000
SCORED_CELLS = KEPT_CELLS // 2

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
    cells = archive.cells.items()
    scored = sorted(cells, key=lambda item: (item[1].score, -item[1].visits), reverse=True)
    kept = dict(scored[:SCORED_CELLS])
    others = [item for item in cells if item[0] not in kept]
    rare = sorted(others, key=lambda item: (item[1].novelty, -item[1].visits), reverse=True)
    kept.update(rare[: KEPT_CELLS - len(kept)])
    return Archive(kept, archive.novelty)


def merged(archives: list[Archive]) -> Archive:
    cells: dict[CellKey, Cell] = {}
    for archive in archives:
        for key, cell in archive.cells.items():
            if key not in cells or cell.frames < cells[key].frames:
                cells[key] = cell
    seen = np.logical_or.reduce([archive.novelty.seen for archive in archives])
    return pruned(Archive(cells, novelty_of(seen)))


def archive_key(start: Start, bound: int) -> ArchiveKey:
    return hashlib.sha256(b"|".join([start.state.encode(), start.emulator_state, str(bound).encode()])).hexdigest()


class Archives:
    def __init__(self, directory: Path, files: int) -> None:
        self.directory = directory
        self.files = files
        self.kept: LRUCache[ArchiveKey, Archive] = LRUCache(maxsize=KEPT_ARCHIVES)
        directory.mkdir(parents=True, exist_ok=True)

    def resumed(self, runner: SegmentRunner, start: Start, bound: int) -> Archive:
        key = archive_key(start, bound)
        if key in self.kept:
            return self.kept[key]
        single = self.directory / f"{key}{ARCHIVE_SUFFIX}"
        saved = [*sorted((self.directory / key).glob(f"*{ARCHIVE_SUFFIX}")), *([single] if single.is_file() else [])]
        if saved:
            return merged([Archive(*load_archive(path)) for path in saved])
        return fresh_archive(runner, start)

    def keep(self, start: Start, bound: int, archive: Archive) -> None:
        key = archive_key(start, bound)
        kept = pruned(archive)
        self.kept[key] = kept
        folder = self.directory / key
        folder.mkdir(exist_ok=True)
        save_archive(folder / f"{os.getpid()}{ARCHIVE_SUFFIX}", start.state, kept.cells, kept.novelty)
        (self.directory / f"{key}{ARCHIVE_SUFFIX}").unlink(missing_ok=True)
        for path in newest(folder.glob(f"*{ARCHIVE_SUFFIX}"))[self.files :]:
            path.unlink(missing_ok=True)
        for folder in newest(path for path in self.directory.iterdir() if path.is_dir())[KEPT_ARCHIVES:]:
            shutil.rmtree(folder, ignore_errors=True)
