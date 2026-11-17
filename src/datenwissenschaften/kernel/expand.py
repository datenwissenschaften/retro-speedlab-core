from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

STAY, LEAVE, FAIL, BROKEN = 0, 1, 2, 3
BYTE = 0xFF

Bytes = NDArray[np.uint8]
Wide = NDArray[np.int32]
Counts = NDArray[np.int64]


def expander(step: Any, prime: Any, judge: Any) -> Callable[..., None]:
    def expand(
        index: int,
        rams: Bytes,
        macros: Wide,
        lengths: Counts,
        phase: int,
        data: Counts,
        wide: Wide,
        found: Bytes,
        memory: Counts,
        scratch: Counts,
        rewards: Counts,
        outcomes: Counts,
        frames: Counts,
    ) -> None:
        moves = lengths.shape[0]
        node, move = index // moves, index % moves
        ram = wide[index]
        for address in range(ram.shape[0]):
            ram[address] = rams[node, address]
        prime(ram, memory[index], scratch[index], data)
        reward, outcome, played = 0, STAY, 0
        while outcome == STAY and played < lengths[move]:
            step(ram, macros[move, played], data)
            gained, outcome = judge(ram, phase, memory[index], scratch[index], data)
            reward += gained
            played += 1
        for address in range(ram.shape[0]):
            outcome = BROKEN if ram[address] < 0 or ram[address] > BYTE else outcome
            found[index, address] = ram[address]
        rewards[index], outcomes[index], frames[index] = reward, outcome, played

    return expand
