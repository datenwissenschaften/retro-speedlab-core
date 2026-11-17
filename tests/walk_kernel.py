import numpy as np
from numpy.typing import NDArray

from datenwissenschaften.kernel.expand import LEAVE, STAY

X, WON, CHEER = 0, 1, 2
LEFT, RIGHT = 6, 7
GOAL_X = 200
MEMORY, SCRATCH = 1, 1
DATA = np.zeros(1, np.int64)
WALKED = 0

Ram = NDArray[np.int32]
Counts = NDArray[np.int64]


def step(ram: Ram, buttons: Ram, data: Counts) -> None:
    ram[CHEER] = 0
    if ram[WON]:
        return
    ram[X] = max(0, min(GOAL_X, ram[X] + buttons[RIGHT] - buttons[LEFT]))
    ram[WON] = 1 if ram[X] == GOAL_X else 0
    ram[CHEER] = ram[WON]


def prime(ram: Ram, memory: Counts, scratch: Counts, data: Counts) -> None:
    memory[WALKED] = ram[X]


def judge(ram: Ram, phase: int, memory: Counts, scratch: Counts, data: Counts) -> tuple[int, int]:
    gained = max(0, int(ram[X]) - int(memory[WALKED]))
    memory[WALKED] = ram[X]
    return gained, LEAVE if ram[WON] else STAY
