import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import msgspec

from datenwissenschaften.route.position import Position
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Spot, key_spot
from datenwissenschaften.ui.media import jpeg

if TYPE_CHECKING:
    from datenwissenschaften.tas.search import Cell, Crossing

LIVE_DIR = "live"
ATTEMPT_FILE = "attempt.jpg"
REPORT_SUFFIX = ".msgpack"
REPORT_SECONDS = 2.0
FRAME_SECONDS = 0.5
MAP_CELLS = 2000


@dataclass(slots=True, frozen=True)
class Watch:
    directory: Path
    shows: bool


class WorkerReport(msgspec.Struct, frozen=True):
    seed: int
    shows: bool
    cells: int
    frames: int
    best_frames: int | None
    visits: list[tuple[int, int, int, int]]
    ram: dict[str, int]
    position: Position | None


def visits(archive: "dict[CellKey, Cell]") -> list[tuple[int, int, int, int]]:
    counted: Counter[Spot] = Counter()
    for key, cell in archive.items():
        if (found := key_spot(key)) is not None:
            counted[found] += 1 + cell.visits
    return [(*found, count) for found, count in counted.most_common(MAP_CELLS)]


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
    def __init__(self, watch: Watch, seed: int) -> None:
        self.watch = watch
        self.seed = seed
        self.frames = 0
        self.report_due = 0.0
        self.frame_due = 0.0 if watch.shows else math.inf

    def step(
        self, runner: SegmentRunner, archive: "dict[CellKey, Cell]", crossings: "list[Crossing]", frames: int
    ) -> None:
        self.frames += frames
        now = time.monotonic()
        if now >= self.frame_due:
            self.frame_due = now + FRAME_SECONDS
            atomic_write(self.watch.directory / ATTEMPT_FILE, jpeg(runner.emulator.em.get_screen()))
        if now >= self.report_due:
            self.report_due = now + REPORT_SECONDS
            report = self.report(runner, archive, crossings)
            atomic_write(self.watch.directory / f"{self.seed}{REPORT_SUFFIX}", msgspec.msgpack.encode(report))

    def report(
        self, runner: SegmentRunner, archive: "dict[CellKey, Cell]", crossings: "list[Crossing]"
    ) -> WorkerReport:
        best = crossings[0].frames if crossings else None
        ram = runner.ram
        return WorkerReport(
            self.seed,
            self.watch.shows,
            len(archive),
            self.frames,
            best,
            visits(archive),
            ram.describe(),
            ram.position(),
        )
