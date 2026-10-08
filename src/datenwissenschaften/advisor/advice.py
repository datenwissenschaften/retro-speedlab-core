from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class Advice:
    action: int
    probabilities: dict[str, float]


@dataclass(slots=True, frozen=True)
class AdvisorDecision:
    action: int
    behavior_probability: float
    value: float
    inputs: np.ndarray
