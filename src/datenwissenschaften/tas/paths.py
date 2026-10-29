import os
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

TAS_DIR = "tas"
TAS_MOVIE = "tas.bk2"
PROGRESS_FILE = "progress.msgpack"


def tas_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / TAS_DIR


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
