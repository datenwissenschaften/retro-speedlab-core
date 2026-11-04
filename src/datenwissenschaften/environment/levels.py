from collections.abc import Sequence
from typing import Any

from datenwissenschaften.states.state import State

Levels = tuple[tuple[str, tuple[type[State[Any]], ...]], ...]


def level_map(levels: Levels, state_classes: Sequence[type[State[Any]]]) -> dict[str, tuple[str, ...]]:
    names = [state_cls.__name__ for state_cls in state_classes]
    mapped: dict[str, tuple[str, ...]] = {}
    for level, classes in levels:
        members = tuple(state_cls.__name__ for state_cls in classes)
        if level in names:
            raise ValueError(f"Level {level} has the name of a state")
        if not members or any(member not in names for member in members):
            raise ValueError(f"Every state of {level} must be one of state_classes")
        start = names.index(members[0])
        if tuple(names[start : start + len(members)]) != members:
            raise ValueError(f"The states of {level} must follow each other in state_classes")
        mapped[level] = members
    return mapped
