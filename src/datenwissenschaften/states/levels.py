from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State


def current_level(ram: RamInfo) -> int:
    level = ram.level()
    if level is None:
        raise RuntimeError(f"{type(ram).__name__}.level() reads no level, so the level states cannot tell levels apart")
    return level


class LevelState(State[RamInfo]):
    goal: str
    following: type[State[RamInfo]] | None

    def _on_reset(self) -> None:
        self._entry, self._progress = current_level(self.ram), self.ram.progress()

    def _reward(self) -> float:
        gained, self._progress = self.ram.progress() - self._progress, self.ram.progress()
        return float(max(0, gained))

    def _exited(self) -> bool:
        return current_level(self.ram) == self._entry + 1

    def _terminated(self) -> bool:
        return current_level(self.ram) not in (self._entry, self._entry + 1)

    def _won(self) -> bool:
        return self.following is None and self._exited()

    def _next(self) -> type[State[RamInfo]] | None:
        return self.following if self.following is not None and self._exited() else None


def level_states(first: int, last: int) -> tuple[type[LevelState], ...]:
    following: type[LevelState] | None = None
    states: list[type[LevelState]] = []
    for number in range(last, first - 1, -1):
        following = type(f"Level{number}", (LevelState,), {"goal": f"Beat level {number}", "following": following})
        states.insert(0, following)
    return tuple(states)
