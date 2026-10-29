from typing import Generic, TypeVar

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)


class StateMachine(Generic[T]):
    def __init__(self, start_state_cls: type[State[T]]) -> None:
        self.start_state = start_state_cls()
        self.current_state = self.start_state
        self.last_transition: tuple[str, str] | None = None
        self.states_by_type: dict[type[State[T]], State[T]] = {start_state_cls: self.start_state}

    @property
    def state_name(self) -> str:
        return self.current_state.__class__.__name__

    def reset(self, ram: T, state_type: type[State[T]] | None) -> None:
        self.current_state = self.start_state if state_type is None else self._state(state_type)
        self.last_transition = None
        self.current_state.reset(ram)

    def step(self, ram: T) -> tuple[float, bool, bool]:
        reward, terminated, truncated, next_state_type = self.current_state.step(ram)
        self.last_transition = None
        if next_state_type is not None:
            previous_state_name = self.state_name
            self.current_state = self._state(next_state_type)
            self.current_state.reset(ram)
            self.last_transition = (previous_state_name, self.state_name)
        return reward, terminated, truncated

    def _state(self, state_cls: type[State[T]]) -> State[T]:
        if state_cls not in self.states_by_type:
            self.states_by_type[state_cls] = state_cls()
        return self.states_by_type[state_cls]
