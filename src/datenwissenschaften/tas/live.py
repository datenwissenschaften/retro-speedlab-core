import time
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from datenwissenschaften.tas.mosaic import TILE_SUFFIX, tile
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.runner import CellKey, SegmentRunner

if TYPE_CHECKING:
    from datenwissenschaften.tas.search import Cell, Crossing

LIVE_DIR = "live"
REPORT_SUFFIX = ".msgpack"
REPORT_SECONDS = 2.0
FRAME_SECONDS = 1.0


class WorkerReport(msgspec.Struct, frozen=True):
    seed: int
    cells: int
    frames: int
    best_frames: int | None


def read_reports(directory: Path) -> list[WorkerReport]:
    reports = []
    for path in sorted(directory.glob(f"*{REPORT_SUFFIX}")):
        try:
            reports.append(msgspec.msgpack.decode(path.read_bytes(), type=WorkerReport))
        except FileNotFoundError:
            continue
    return sorted(reports, key=lambda report: report.seed)


def clear_reports(directory: Path) -> None:
    for path in directory.glob("*"):
        path.unlink(missing_ok=True)


class WorkerLive:
    def __init__(self, directory: Path, seed: int) -> None:
        self.directory = directory
        self.seed = seed
        self.frames = 0
        self.report_due = 0.0
        self.frame_due = 0.0

    def step(
        self, runner: SegmentRunner, archive: "dict[CellKey, Cell]", crossings: "list[Crossing]", frames: int
    ) -> None:
        self.frames += frames
        now = time.monotonic()
        if now >= self.frame_due:
            self.frame_due = now + FRAME_SECONDS
            atomic_write(self.directory / f"{self.seed}{TILE_SUFFIX}", tile(runner.emulator.em.get_screen()))
        if now >= self.report_due:
            self.report_due = now + REPORT_SECONDS
            best = crossings[0].frames if crossings else None
            report = WorkerReport(self.seed, len(archive), self.frames, best)
            atomic_write(self.directory / f"{self.seed}{REPORT_SUFFIX}", msgspec.msgpack.encode(report))
