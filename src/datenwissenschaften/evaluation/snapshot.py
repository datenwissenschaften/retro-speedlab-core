import shutil
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

SNAPSHOT_STAMP = "%Y%m%dT%H%M%S"


def snapshot_models(config: RetroSpeedlabConfig) -> Path:
    identity = config.training.game_identity
    source = config.paths.models_dir / identity
    source.mkdir(parents=True, exist_ok=True)
    snapshot = config.paths.cache_dir / "ab" / identity / f"{datetime.now(UTC):{SNAPSHOT_STAMP}}"
    shutil.copytree(source, snapshot / identity)
    return snapshot


def with_models(config: RetroSpeedlabConfig, models_dir: Path) -> RetroSpeedlabConfig:
    return replace(config, paths=replace(config.paths, models_dir=models_dir))
