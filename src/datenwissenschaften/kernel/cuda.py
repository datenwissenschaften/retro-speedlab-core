from collections.abc import Callable
from types import FunctionType
from typing import Any

import numpy as np
from numba_cuda_mlir import cuda as typed_cuda

from datenwissenschaften.kernel.expand import Bytes, Counts, Wide

Launched = tuple[Bytes, Counts, Counts, Counts]
Launch = Callable[[Bytes, Wide, Counts, int, Any, int, int], Launched]
THREADS = 128
cuda: Any = typed_cuda


def jit(function: FunctionType) -> Any:
    return cuda.jit(device=True)(function)


def resident(data: Counts) -> Any:
    return cuda.to_device(data)


def launcher(expand_function: Callable[..., None]) -> Launch:
    expand = cuda.jit(device=True)(expand_function)

    @cuda.jit
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
        index = cuda.grid(1)
        if index < wide.shape[0]:
            expand(index, rams, macros, lengths, phase, data, wide, found, memory, scratch, rewards, outcomes, frames)

    def launch(
        rams: Bytes, macros: Wide, lengths: Counts, phase: int, data: Any, memory: int, scratch: int
    ) -> Launched:
        count, size = rams.shape[0] * lengths.shape[0], rams.shape[1]
        wide, found = cuda.device_array((count, size), np.int32), cuda.device_array((count, size), np.uint8)
        results = [cuda.device_array(count, np.int64) for _ in range(3)]
        working = (cuda.device_array((count, memory), np.int64), cuda.device_array((count, scratch), np.int64))
        inputs = (cuda.to_device(rams), cuda.to_device(macros), cuda.to_device(lengths))
        run[(count + THREADS - 1) // THREADS, THREADS](*inputs, phase, data, wide, found, *working, *results)
        rewards, outcomes, frames = (result.copy_to_host() for result in results)
        return found.copy_to_host(), rewards, outcomes, frames

    return launch
