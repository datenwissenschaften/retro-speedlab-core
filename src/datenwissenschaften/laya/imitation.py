from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from datenwissenschaften.advisor.inputs import expand
from datenwissenschaften.laya.heads import PolicyHead

IMITATION_WEIGHT = 1.0
MASTERED_IMITATION_WEIGHT = 0.1


class Shown(Protocol):
    @property
    def state(self) -> str: ...

    @property
    def question(self) -> str: ...

    @property
    def action(self) -> int: ...


@dataclass(slots=True, frozen=True)
class DemonstrationStep:
    state: str
    question: str
    action: int
    ram: np.ndarray
    slots: np.ndarray

    @property
    def inputs(self) -> np.ndarray:
        return expand(self.ram, self.slots)


@dataclass(slots=True, frozen=True)
class Example:
    state: str
    question: str
    action: int


@dataclass(slots=True, frozen=True)
class Lesson[T]:
    examples: list[T]
    weight: float


def imitation_loss(policy: PolicyHead, options: torch.Tensor, actions: torch.Tensor, weight: float) -> torch.Tensor:
    log_probs = torch.log_softmax(policy(options), -1)
    return -weight * log_probs.gather(1, actions[:, None]).mean()
