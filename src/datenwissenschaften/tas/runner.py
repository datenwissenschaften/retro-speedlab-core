from dataclasses import dataclass

import numpy as np

from datenwissenschaften.environment.recording import restore_emulator_state
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper, lost_life, state_class
from datenwissenschaften.training.trainer import phases

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


class SegmentRunner:
    def __init__(self, wrapper: StateMachineGymWrapper) -> None:
        self.wrapper = wrapper
        self.emulator = wrapper.env.unwrapped
        self.order = phases(type(wrapper))
        self.classes = (wrapper.start_state_cls, *wrapper.state_classes)
        wrapper.env.reset()
        self.ram = wrapper.read_ram()

    def power_on(self) -> Start:
        self.wrapper.env.reset()
        self.ram = self.wrapper.read_ram()
        self.wrapper.state_machine.reset(self.ram, self.emulator.em.get_screen(), None)
        return Start(self.order[0], self.snapshot())

    def restore(self, start: Start) -> None:
        restore_emulator_state(self.emulator, start.emulator_state)
        self.ram = self.wrapper.read_ram()
        state_type = None if start.state == self.order[0] else state_class(self.classes, start.state)
        self.wrapper.state_machine.reset(self.ram, self.emulator.em.get_screen(), state_type)

    def decide(self, action: int) -> Outcome:
        machine = self.wrapper.state_machine
        pressed, alive, won = [], True, False
        for buttons in self.wrapper.current_action_table()[action]:
            frame, *_ = self.wrapper.env.step(buttons)
            pressed.append(buttons)
            ram = self.wrapper.read_ram()
            _, terminated, truncated = machine.step(ram, frame)
            won = machine.current_state._won()
            alive = won or not (terminated or truncated or lost_life(self.ram, ram))
            self.ram = ram
            if won or not alive or machine.last_transition is not None:
                break
        return Outcome(np.asarray(pressed, dtype=np.int8), machine.state_name, alive, won)

    def forward(self, origin: str, reached: str) -> bool:
        return self.order.index(reached) > self.order.index(origin)

    def snapshot(self) -> bytes:
        return bytes(self.emulator.em.get_state())

    def key(self, frames: int) -> CellKey:
        state, position = self.wrapper.state_machine.state_name, self.ram.position()
        if position is None:
            return (state, frames // TIME_BUCKET_FRAMES)
        cell = (position.x // CELL_PIXELS, position.y // CELL_PIXELS, position.height // CELL_PIXELS)
        return (state, position.area, *cell, position.airborne, self.ram.dynamics())
