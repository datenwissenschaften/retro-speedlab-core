from pathlib import Path

import msgspec
import numpy as np

from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.runner import Start

POLICY_START_FILE = "policy_start.msgpack"
POLICY_EXITS_FILE = "policy_exits.msgpack"

PolicyExits = dict[str, list[list[int]]]


def save_policy_start(path: Path, start: Start) -> None:
    atomic_write(path, msgspec.msgpack.encode(start))


def load_policy_start(path: Path) -> Start | None:
    return msgspec.msgpack.decode(path.read_bytes(), type=Start) if path.is_file() else None


def load_policy_exits(path: Path) -> PolicyExits:
    return msgspec.msgpack.decode(path.read_bytes(), type=PolicyExits) if path.is_file() else {}


def save_policy_exit(path: Path, state: str, pressed: np.ndarray) -> bool:
    exits = load_policy_exits(path)
    if state in exits and len(exits[state]) <= len(pressed):
        return False
    atomic_write(path, msgspec.msgpack.encode({**exits, state: pressed.tolist()}))
    return True
