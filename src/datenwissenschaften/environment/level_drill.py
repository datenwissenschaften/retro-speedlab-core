from collections import Counter
from pathlib import Path

import msgspec
from loguru import logger

from datenwissenschaften.curriculum import atomic_write
from datenwissenschaften.environment.levels import LevelTargets
from datenwissenschaften.ui.telemetry import publish_metadata

ENDINGS_WINDOW = 50
MIN_ENDINGS = 20
DRILL_WINDOW = 10
RELIABLE_PASSES = 3


class LevelDrill:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.level: str | None = None

    def pick(self, frontier: str | None, targets: LevelTargets) -> str | None:
        drilled = self.target(frontier) if targets.is_level(frontier) else None
        self.level = None if drilled is None else frontier
        return drilled or frontier

    def finish(self, state: str, passed: bool) -> None:
        if self.record_drill(self.level, passed):
            logger.info(f"{state} passes reliably again; {self.level} is practised whole")

    def publish(self, levels: dict[str, tuple[str, ...]]) -> None:
        publish_metadata("level_drills", {level: self.progress(level) for level in levels}, replace=True)

    def target(self, level: str) -> str | None:
        endings = self._read(self._endings_path(level), list[str])
        if len(endings) < MIN_ENDINGS:
            return None
        return Counter(endings).most_common(1)[0][0]

    def record_ending(self, level: str, state: str) -> None:
        endings = self._read(self._endings_path(level), list[str])
        atomic_write(self._endings_path(level), msgspec.json.encode([*endings, state][-ENDINGS_WINDOW:]))

    def record_drill(self, level: str, passed: bool) -> bool:
        outcomes = [*self._read(self._drill_path(level), list[bool]), passed][-DRILL_WINDOW:]
        if sum(outcomes) >= RELIABLE_PASSES:
            self._drill_path(level).unlink(missing_ok=True)
            self._endings_path(level).unlink(missing_ok=True)
            return True
        atomic_write(self._drill_path(level), msgspec.json.encode(outcomes))
        return False

    def progress(self, level: str) -> dict[str, object]:
        outcomes = self._read(self._drill_path(level), list[bool])
        return {
            "endings": dict(Counter(self._read(self._endings_path(level), list[str]))),
            "drill": self.target(level),
            "drill_passes": sum(outcomes),
            "drill_attempts": len(outcomes),
        }

    @staticmethod
    def _read[T](path: Path, kind: type[T]) -> T:
        return msgspec.json.decode(path.read_bytes(), type=kind) if path.is_file() else kind()

    def _endings_path(self, level: str) -> Path:
        return self.root / f"{level}.endings.json"

    def _drill_path(self, level: str) -> Path:
        return self.root / f"{level}.drill.json"
