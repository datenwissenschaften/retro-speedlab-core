from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from datenwissenschaften.environment.recording import restore_emulator_state
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper, known_lives, lost_life, phases, state_class
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.route.route_map import CELL_SIZE, Needs
from datenwissenschaften.route.route_map import Cell as RouteCell

CELL_PIXELS = 8
TIME_BUCKET_FRAMES = 30
CONTEXT_PIXELS = 16
POSITIONED_KEY = 7
ROUTE_SCALE = CELL_SIZE // CELL_PIXELS

CellKey = tuple[object, ...]


def route_cell(key: CellKey) -> RouteCell | None:
    if len(key) != POSITIONED_KEY or not isinstance(key[5], bool) or not isinstance(key[6], tuple):
        return None
    area, x, y = key[1], key[2], key[3]
    if not isinstance(area, int) or not isinstance(x, int) or not isinstance(y, int):
        return None
    return area, x // ROUTE_SCALE, y // ROUTE_SCALE


def route_needs(key: CellKey) -> Needs:
    dynamics = key[6] if len(key) == POSITIONED_KEY else None
    if not isinstance(dynamics, tuple):
        return ()
    return tuple(value for value in cast(tuple[object, ...], dynamics) if isinstance(value, int))


@dataclass(slots=True, frozen=True)
class Start:
    state: str
    emulator_state: bytes


@dataclass(slots=True, frozen=True)
class Outcome:
    pressed: np.ndarray
    state: str
    alive: bool
    won: bool
    reward: float


class SegmentRunner:
    def __init__(self, wrapper: StateMachineGymWrapper[RamInfo]) -> None:
        self.wrapper = wrapper
        self.emulator = wrapper.env
        self.order = phases(type(wrapper))
        self.classes = (wrapper.start_state_cls, *wrapper.state_classes)
        self.emulator.reset()
        self.ram = wrapper.read_ram()
        self.lives = self.ram.remaining_lives()

    def power_on(self) -> Start:
        self.emulator.reset()
        self.ram = self.wrapper.read_ram()
        self.lives = self.ram.remaining_lives()
        self.wrapper.state_machine.reset(self.ram, None)
        return Start(self.order[0], self.snapshot())

    def restore(self, start: Start) -> None:
        restore_emulator_state(self.emulator, start.emulator_state)
        self.ram = self.wrapper.read_ram()
        self.lives = self.ram.remaining_lives()
        state_type = None if start.state == self.order[0] else state_class(self.classes, start.state)
        self.wrapper.state_machine.reset(self.ram, state_type)

    def press(self, buttons: np.ndarray) -> tuple[bool, bool, float]:
        for player, mask in enumerate(self.emulator.action_to_array(buttons)):
            self.emulator.em.set_button_mask(mask, player)
        self.emulator.em.step()
        self.emulator.data.update_ram()
        ram = self.wrapper.read_ram()
        machine = self.wrapper.state_machine
        reward, terminated, truncated = machine.step(ram)
        won = machine.current_state.won()
        alive = won or not (terminated or truncated or lost_life(self.lives, ram))
        self.ram, self.lives = ram, known_lives(self.lives, ram)
        return alive, won, reward

    def decide(self, action: int) -> Outcome:
        machine = self.wrapper.state_machine
        pressed: list[NDArray[np.int8]] = []
        alive, won, reward = True, False, 0.0
        for buttons in self.wrapper.current_action_table()[action]:
            pressed.append(buttons)
            alive, won, frame_reward = self.press(buttons)
            reward += frame_reward
            if won or not alive or machine.last_transition is not None:
                break
        return Outcome(np.asarray(pressed, dtype=np.int8), machine.state_name, alive, won, reward)

    def forward(self, origin: str, reached: str) -> bool:
        return self.order.index(reached) > self.order.index(origin)

    def set_level(self, level: int) -> None:
        if self.wrapper.level_address is None:
            raise RuntimeError(f"{type(self.wrapper).__name__} names no level_address to load a level with")
        self.emulator.data.memory.assign(self.wrapper.level_address, "|u1", level % 256)

    def memory(self) -> np.ndarray:
        return self.emulator.get_ram()

    def context(self) -> tuple[str, str]:
        state, position = self.wrapper.state_machine.state_name, self.ram.position()
        if position is None:
            return (state, state)
        place = (position.area, position.x // CONTEXT_PIXELS, position.y // CONTEXT_PIXELS)
        return (f"{state}|{'|'.join(map(str, place))}", state)

    def snapshot(self) -> bytes:
        return bytes(self.emulator.em.get_state())

    def remaining(self) -> int | None:
        return self.wrapper.routes.remaining(self.ram.position(), self.ram.dynamics())

    def key(self, frames: int) -> CellKey:
        state, position = self.wrapper.state_machine.state_name, self.ram.position()
        if position is None:
            return (state, *self.ram.dynamics()) if self.ram.dynamics() else (state, frames // TIME_BUCKET_FRAMES)
        cell = (position.x // CELL_PIXELS, position.y // CELL_PIXELS, position.height // CELL_PIXELS)
        return (state, position.area, *cell, position.airborne, self.ram.dynamics())
