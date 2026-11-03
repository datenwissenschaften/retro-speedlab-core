from datetime import UTC, datetime
from pathlib import Path

import msgspec

from datenwissenschaften.tas.paths import atomic_write

STUCK_FILE = "stuck.json"
FAILURES_FILE = "failures.json"

Failures = dict[str, int]


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


def load_failures(path: Path) -> Failures:
    return msgspec.json.decode(path.read_bytes(), type=Failures) if path.is_file() else {}


def failed(path: Path, state: str) -> int:
    failures = load_failures(path)
    failures[state] = failures.get(state, 0) + 1
    atomic_write(path, msgspec.json.encode(failures))
    return failures[state]


def cleared(path: Path, state: str) -> None:
    failures = load_failures(path)
    if state in failures:
        del failures[state]
        atomic_write(path, msgspec.json.encode(failures))
