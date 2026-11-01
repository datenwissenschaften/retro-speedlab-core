from dataclasses import dataclass

import numpy as np

from datenwissenschaften.environment.recording import restore_emulator_state
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper, lost_life, phases, state_class

CELL_PIXELS = 8
TIME_BUCKET_FRAMES = 30

CellKey = tuple[object, ...]


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
    def __init__(self, wrapper: StateMachineGymWrapper) -> None:
        self.wrapper = wrapper
        self.emulator = wrapper.env
        self.order = phases(type(wrapper))
        self.classes = (wrapper.start_state_cls, *wrapper.state_classes)
        self.emulator.reset()
        self.ram = wrapper.read_ram()

    def power_on(self) -> Start:
        self.emulator.reset()
        self.ram = self.wrapper.read_ram()
        self.wrapper.state_machine.reset(self.ram, None)
        return Start(self.order[0], self.snapshot())

    def restore(self, start: Start) -> None:
        restore_emulator_state(self.emulator, start.emulator_state)
        self.ram = self.wrapper.read_ram()
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
        won = machine.current_state._won()
        alive = won or not (terminated or truncated or lost_life(self.ram, ram))
        self.ram = ram
        return alive, won, reward

    def decide(self, action: int) -> Outcome:
        machine = self.wrapper.state_machine
        pressed, alive, won, reward = [], True, False, 0.0
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

    def snapshot(self) -> bytes:
        return bytes(self.emulator.em.get_state())

    def remaining(self) -> int | None:
        return self.wrapper.routes.remaining(self.ram.position())

    def key(self, frames: int) -> CellKey:
        state, position = self.wrapper.state_machine.state_name, self.ram.position()
        if position is None:
            return (state, *self.ram.dynamics()) if self.ram.dynamics() else (state, frames // TIME_BUCKET_FRAMES)
        cell = (position.x // CELL_PIXELS, position.y // CELL_PIXELS, position.height // CELL_PIXELS)
        return (state, position.area, *cell, position.airborne, self.ram.dynamics())
