from collections.abc import Callable
from types import FunctionType
from typing import Any

import numpy as np
from numba import njit, prange

from datenwissenschaften.kernel.expand import Bytes, Counts, Wide

Launched = tuple[Bytes, Counts, Counts, Counts]
Launch = Callable[[Bytes, Wide, Counts, int, Counts, int, int], Launched]


def jit(function: FunctionType) -> Any:
    return njit(function)


def resident(data: Counts) -> Counts:
    return data


def launcher(expand_function: Callable[..., None]) -> Launch:
    expand = njit(expand_function)

    @njit(parallel=True)
    def run(
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
        for index in prange(wide.shape[0]):
            expand(index, rams, macros, lengths, phase, data, wide, found, memory, scratch, rewards, outcomes, frames)

    def launch(
        rams: Bytes, macros: Wide, lengths: Counts, phase: int, data: Counts, memory: int, scratch: int
    ) -> Launched:
        count = rams.shape[0] * lengths.shape[0]
        wide, found = np.empty((count, rams.shape[1]), np.int32), np.empty((count, rams.shape[1]), np.uint8)
        rewards, outcomes, frames = (np.empty(count, np.int64) for _ in range(3))
        working = (np.empty((count, memory), np.int64), np.empty((count, scratch), np.int64))
        run(rams, macros, lengths, phase, data, wide, found, *working, rewards, outcomes, frames)
        return found, rewards, outcomes, frames

    return launch
