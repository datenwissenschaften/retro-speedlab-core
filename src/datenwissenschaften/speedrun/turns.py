from pathlib import Path

import msgspec

from datenwissenschaften.speedrun.paths import atomic_write

TURN_FILE = "turn.json"
COURSE_FILE = "course.json"
LONGEST_TURN = 4

LevelStates = dict[str, tuple[str, ...]]


class Turn(msgspec.Struct, frozen=True):
    part: str
    searches: int
    found_at: int = 0

    def over(self, patience: int) -> bool:
        return self.searches - self.found_at >= patience or self.searches >= patience * LONGEST_TURN

    def searched(self) -> "Turn":
        return Turn(self.part, self.searches + 1, self.found_at)

    def found(self) -> "Turn":
        return Turn(self.part, self.searches, self.searches)


def load_turn(path: Path) -> Turn | None:
    return msgspec.json.decode(path.read_bytes(), type=Turn) if path.is_file() else None


def save_turn(path: Path, turn: Turn) -> None:
    atomic_write(path, msgspec.json.encode(turn))


def turn_of(path: Path, part: str, patience: int) -> Turn:
    turn = load_turn(path)
    if turn is None or turn.part != part or turn.over(patience):
        return Turn(part, 0)
    return turn


def changed_level(path: Path, levels: LevelStates) -> str | None:
    before = msgspec.json.decode(path.read_bytes(), type=LevelStates) if path.is_file() else levels
    atomic_write(path, msgspec.json.encode(levels))
    return next((level for level, states in levels.items() if level not in before or before[level] != states), None)
