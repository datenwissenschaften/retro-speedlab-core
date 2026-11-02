from pathlib import Path

import msgspec

from datenwissenschaften.tas.paths import atomic_write

TURN_FILE = "turn.json"


class Turn(msgspec.Struct, frozen=True):
    part: str
    searches: int


def load_turn(path: Path) -> Turn | None:
    return msgspec.json.decode(path.read_bytes(), type=Turn) if path.is_file() else None


def save_turn(path: Path, turn: Turn) -> None:
    atomic_write(path, msgspec.json.encode(turn))


def searches_taken(path: Path, part: str, patience: int) -> int:
    turn = load_turn(path)
    if turn is None or turn.part != part or turn.searches >= patience:
        return 0
    return turn.searches
