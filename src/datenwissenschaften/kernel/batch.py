from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from types import ModuleType

import numpy as np

from datenwissenschaften.kernel.compiler import compiled
from datenwissenschaften.kernel.expand import BROKEN, Bytes, Counts, Wide, expander

KERNEL = "speedrun.kernels.{slug}.kernel"
DEVICES = "datenwissenschaften.kernel.{device}"


class BeamKernel(StrEnum):
    CARTRIDGE = "cartridge"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(slots=True, frozen=True)
class Expanded:
    rams: Bytes
    memories: Counts
    rewards: Counts
    outcomes: Counts
    frames: Counts


class Batch:
    def __init__(self, kernel: ModuleType, device: BeamKernel) -> None:
        if device == BeamKernel.CARTRIDGE:
            raise ValueError("The cartridge steps one game at a time; a batch runs on the cpu or cuda kernel")
        target = import_module(DEVICES.format(device=device))
        functions = compiled(kernel, target.jit)
        self.memory, self.scratch = int(kernel.MEMORY), int(kernel.SCRATCH)
        self.data = target.resident(np.asarray(kernel.DATA, np.int64))
        self.launch = target.launcher(expander(functions["step"], functions["prime"], functions["judge"]))

    def expand(self, rams: Bytes, carried: Counts, macros: Wide, lengths: Counts, phase: int) -> Expanded:
        found, memories, rewards, outcomes, frames = self.launch(
            rams, carried, macros, lengths, phase, self.data, self.memory, self.scratch
        )
        if (outcomes == BROKEN).any():
            raise RuntimeError("The game kernel wrote a value outside a byte into RAM")
        return Expanded(found, memories, rewards, outcomes, frames)


def game_kernel(slug: str) -> ModuleType:
    return import_module(KERNEL.format(slug=slug))
