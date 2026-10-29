from typing import Generic, TypeVar

import numpy as np
from stable_retro import RetroEnv

from datenwissenschaften.environment.levels import Levels, level_map
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)

ACTION_TABLE_DIMENSIONS = 3


def state_class(state_classes: tuple[type[State[T]], ...], state_name: str) -> type[State[T]]:
    for state_cls in state_classes:
        if state_cls.__name__ == state_name:
            return state_cls
    raise ValueError(f"Unknown state: {state_name}")


def phases(wrapper_cls: type["StateMachineGymWrapper"]) -> tuple[str, ...]:
    classes = (wrapper_cls.start_state_cls, *wrapper_cls.state_classes)
    return tuple(dict.fromkeys(state_cls.__name__ for state_cls in classes))


def lost_life(before: RamInfo, after: RamInfo) -> bool:
    lives_before, lives_after = before.remaining_lives(), after.remaining_lives()
    return lives_before is not None and lives_after is not None and lives_after < lives_before


class StateMachineGymWrapper(Generic[T]):
    start_state_cls: type[State[T]]
    state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_table: np.ndarray
    levels: Levels

    def __init__(self, env: RetroEnv, routes: RouteMap) -> None:
        if self.action_table.ndim != ACTION_TABLE_DIMENSIONS:
            raise ValueError("action_table must have the shape (actions, frames, buttons).")
        level_map(self.levels, self.state_classes)
        self.env = env
        self.routes = routes
        self.state_machine = StateMachine[T](self.start_state_cls)

    def current_action_table(self) -> np.ndarray:
        return self.action_table

    def read_ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.get_ram())
