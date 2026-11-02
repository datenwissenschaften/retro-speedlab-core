from pathlib import Path

import msgspec

from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.runner import Start

POLICY_START_FILE = "policy_start.msgpack"


def save_policy_start(path: Path, start: Start) -> None:
    atomic_write(path, msgspec.msgpack.encode(start))


def load_policy_start(path: Path) -> Start | None:
    return msgspec.msgpack.decode(path.read_bytes(), type=Start) if path.is_file() else None
