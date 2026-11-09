import time
from pathlib import Path

import msgspec

from datenwissenschaften.speedrun.mosaic import TILE_SUFFIX, tile
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.runner import SegmentRunner

LIVE_DIR = "live"
REPORT_SUFFIX = ".msgpack"
REPORT_SECONDS = 2.0
FRAME_SECONDS = 1.0
SCOUT = "scout"
BACKTRACKER = "backtracker"
BEAM_SCOUT = "beam-scout"
VISION_SCOUT = "vision-scout"
POLISHER = "polisher"
PPO = "ppo"


class WorkerReport(msgspec.Struct, frozen=True):
    seed: int
    role: str
    mode: str
    cells: int
    frames: int
    best_frames: int | None
    closest: int | None


def read_reports(directory: Path) -> list[WorkerReport]:
    reports: list[WorkerReport] = []
    for path in sorted(directory.glob(f"*{REPORT_SUFFIX}")):
        try:
            reports.append(msgspec.msgpack.decode(path.read_bytes(), type=WorkerReport))
        except FileNotFoundError:
            continue
    return sorted(reports, key=lambda report: report.seed)


def clear_reports(directory: Path) -> None:
    for path in directory.glob("[!.]*"):
        path.unlink(missing_ok=True)


class WorkerLive:
    def __init__(self, directory: Path, seed: int, role: str, mode: str) -> None:
        self.directory = directory
        self.seed = seed
        self.role = role
        self.mode = mode
        self.frames = 0
        self.report_due = 0.0
        self.frame_due = 0.0

    def step(self, runner: SegmentRunner, cells: int, best: int | None, frames: int, closest: int | None) -> None:
        self.frames += frames
        now = time.monotonic()
        if now >= self.frame_due:
            self.frame_due = now + FRAME_SECONDS
            atomic_write(self.directory / f"{self.seed}{TILE_SUFFIX}", tile(runner.emulator.em.get_screen()))
        if now >= self.report_due:
            self.report_due = now + REPORT_SECONDS
            report = WorkerReport(self.seed, self.role, self.mode, cells, self.frames, best, closest)
            atomic_write(self.directory / f"{self.seed}{REPORT_SUFFIX}", msgspec.msgpack.encode(report))
