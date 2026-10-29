from datetime import UTC, datetime
from pathlib import Path

import msgspec

from datenwissenschaften.tas.paths import atomic_write

STUCK_FILE = "stuck.json"


class Stuck(msgspec.Struct, frozen=True):
    state: str
    since: str
    searches: int


def load_stuck(path: Path) -> Stuck | None:
    return msgspec.json.decode(path.read_bytes(), type=Stuck) if path.is_file() else None


def mark_stuck(path: Path, state: str) -> Stuck:
    previous = load_stuck(path)
    if previous is not None and previous.state == state:
        stuck = Stuck(state, previous.since, previous.searches + 1)
    else:
        stuck = Stuck(state, datetime.now(UTC).isoformat(), 1)
    atomic_write(path, msgspec.json.encode(stuck))
    return stuck
