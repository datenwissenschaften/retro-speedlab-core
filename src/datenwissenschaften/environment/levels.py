from collections.abc import Sequence

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.states.state import State

Levels = tuple[tuple[str, tuple[type[State], ...]], ...]


def level_map(levels: Levels, state_classes: Sequence[type[State]]) -> dict[str, tuple[str, ...]]:
    names = [state_cls.__name__ for state_cls in state_classes]
    mapped = {}
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


def curriculum_targets(state_classes: Sequence[type[State]], levels: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    level_after = {members[-1]: level for level, members in levels.items()}
    targets = []
    for state_cls in state_classes:
        targets.append(state_cls.__name__)
        if state_cls.__name__ in level_after:
            targets.append(level_after[state_cls.__name__])
    return tuple(targets)


class LevelTargets:
    def __init__(self, state_names: tuple[str, ...], levels: dict[str, tuple[str, ...]]) -> None:
        self.state_names = state_names
        self.levels = levels

    def is_level(self, target: str | None) -> bool:
        return target in self.levels

    def starts(self, target: str, state: str) -> bool:
        return state == target or (self.is_level(target) and self.levels[target][0] == state)

    def completes(self, target: str, previous_state: str, new_state: str) -> bool:
        if not self.is_level(target):
            return previous_state == target
        members = self.levels[target]
        return previous_state in members and new_state not in members

    def start_checkpoint(self, active_state: str | None, curriculum: ReverseCurriculum) -> str | None:
        if not self.is_level(active_state):
            return curriculum.episode_start_state()
        first = self.state_names.index(self.levels[active_state][0])
        candidates = reversed(self.state_names[1 : first + 1])
        return next((state for state in candidates if curriculum.has_checkpoint(state)), None)
