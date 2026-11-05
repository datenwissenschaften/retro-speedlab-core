from datetime import UTC, datetime
from pathlib import Path

import msgspec

from datenwissenschaften.tas.paths import atomic_write

STUCK_FILE = "stuck.json"
FAILURES_FILE = "failures.json"
BARRED_FILE = "barred.json"

Barred = dict[str, str]


class Stuck(msgspec.Struct, frozen=True):
    state: str
    since: str
    searches: int


Failures = dict[str, Stuck]


def load_stuck(path: Path) -> Stuck | None:
    return msgspec.json.decode(path.read_bytes(), type=Stuck) if path.is_file() else None


def stuck_again(previous: Stuck | None, state: str) -> Stuck:
    if previous is not None and previous.state == state:
        return Stuck(state, previous.since, previous.searches + 1)
    return Stuck(state, datetime.now(UTC).isoformat(), 1)


def save_stuck(path: Path, stuck: Stuck) -> Stuck:
    atomic_write(path, msgspec.json.encode(stuck))
    return stuck


def mark_stuck(path: Path, state: str) -> Stuck:
    return save_stuck(path, stuck_again(load_stuck(path), state))


def load_failures(path: Path) -> Failures:
    if not path.is_file():
        return {}
    failures = msgspec.json.decode(path.read_bytes(), type=dict[str, Stuck | int])
    return {
        state: Stuck(state, datetime.now(UTC).isoformat(), failure) if isinstance(failure, int) else failure
        for state, failure in failures.items()
    }


def failed(path: Path, state: str) -> Stuck:
    failures = load_failures(path)
    failures[state] = stuck_again(failures.get(state), state)
    atomic_write(path, msgspec.json.encode(failures))
    return failures[state]


def load_barred(path: Path) -> Barred:
    return msgspec.json.decode(path.read_bytes(), type=Barred) if path.is_file() else {}


def bar(path: Path, leg: str, state: str) -> None:
    atomic_write(path, msgspec.json.encode({**load_barred(path), leg: state}))


def unbar(path: Path, leg: str, state: str) -> None:
    barred = load_barred(path)
    if leg in barred and barred[leg] == state:
        del barred[leg]
        atomic_write(path, msgspec.json.encode(barred))


def cleared(path: Path, state: str) -> None:
    failures = load_failures(path)
    if state in failures:
        del failures[state]
        atomic_write(path, msgspec.json.encode(failures))
