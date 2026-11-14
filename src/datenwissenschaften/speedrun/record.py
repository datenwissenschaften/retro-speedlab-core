from pathlib import Path

import msgspec
from loguru import logger

from datenwissenschaften.speedrun.movie import Segment, frames
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.routes import finished, still_replaying
from datenwissenschaften.speedrun.runner import SegmentRunner

RECORD_FILE = "record.msgpack"


class StoredRecord(msgspec.Struct, frozen=True):
    version: str
    segments: list[Segment]


def retired_path(path: Path, version: str) -> Path:
    return path.with_name(f"{path.stem}-{version}{path.suffix}")


def stored_record(path: Path) -> StoredRecord | None:
    return msgspec.msgpack.decode(path.read_bytes(), type=StoredRecord) if path.is_file() else None


class Record:
    def __init__(self, path: Path, runner: SegmentRunner, version: str) -> None:
        self.path = path
        self.runner = runner
        self.version = version
        self.segments = self.restored()

    def replayed(self, segments: list[Segment]) -> list[Segment] | None:
        replay = still_replaying(self.runner, self.runner.power_on(), segments)
        return replay if len(replay) == len(segments) and finished(replay, None) else None

    def restored(self) -> list[Segment]:
        stored = stored_record(self.path)
        if stored is None:
            return []
        replay = self.replayed(list(stored.segments))
        if replay is None:
            retired = retired_path(self.path, stored.version)
            self.path.replace(retired)
            logger.warning(
                f"The record of version {stored.version} in {frames(stored.segments)} frames does not replay"
                f" from power-on with version {self.version}; it is kept as {retired.name} and the search starts fresh"
            )
            return []
        if stored.version != self.version:
            logger.info(f"The record of version {stored.version} replays identically with version {self.version}")
        self.save(replay)
        logger.info(
            f"The record from power-on is restored and replays: {len(replay)} pieces in {frames(replay)} frames"
        )
        return replay

    def save(self, segments: list[Segment]) -> None:
        self.segments = segments
        atomic_write(self.path, msgspec.msgpack.encode(StoredRecord(self.version, segments)))

    def offer(self, segments: list[Segment]) -> None:
        if not finished(segments, None) or (self.segments and frames(segments) >= frames(self.segments)):
            return
        replay = self.replayed(segments)
        if replay is None:
            logger.warning(f"A won run in {frames(segments)} frames does not replay from power-on; the record stays")
            return
        was = f" (was {frames(self.segments)})" if self.segments else ""
        self.save(replay)
        logger.info(f"New record from power-on: {len(replay)} pieces in {frames(replay)} frames{was}")

    def shown(self, segments: list[Segment]) -> list[Segment]:
        self.offer(segments)
        return self.segments or segments
