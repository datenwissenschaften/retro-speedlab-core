from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from speedlab_engine.console import Console

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


def phases(wrapper_cls: type["StateMachineGymWrapper[RamInfo]"]) -> tuple[str, ...]:
    classes = (wrapper_cls.start_state_cls, *wrapper_cls.state_classes)
    return tuple(dict.fromkeys(state_cls.__name__ for state_cls in classes))


def lost_life(known: int | None, after: RamInfo) -> bool:
    lives = after.remaining_lives()
    return known is not None and lives is not None and lives < known


def known_lives(known: int | None, ram: RamInfo) -> int | None:
    lives = ram.remaining_lives()
    return known if lives is None else lives


class StateMachineGymWrapper(Generic[T]):
    start_state_cls: ClassVar[type[State[Any]]]
    state_classes: ClassVar[tuple[type[State[Any]], ...]]
    ram_info_cls: type[T]
    action_table: NDArray[np.int8]
    levels: Levels

    def __init__(self, env: Console, routes: RouteMap) -> None:
        if self.action_table.ndim != ACTION_TABLE_DIMENSIONS:
            raise ValueError("action_table must have the shape (actions, frames, buttons).")
        level_map(self.levels, self.state_classes)
        self.ram_info_cls.shown_fields()
        self.env = env
        self.routes = routes
        self.state_machine = StateMachine[T](self.start_state_cls)

    def read_ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.get_ram())
