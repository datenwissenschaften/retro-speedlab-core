from dataclasses import dataclass

import numpy as np

SHOWN_PROBABILITY = 0.1
SHOWN_MOVES = 3
PERCENT = 100


@dataclass(slots=True, frozen=True)
class Advice:
    action: int
    probabilities: dict[str, float]

    def __str__(self) -> str:
        ranked = sorted(self.probabilities.items(), key=lambda item: item[1], reverse=True)[:SHOWN_MOVES]
        shown = [(name, probability) for name, probability in ranked if probability >= SHOWN_PROBABILITY]
        return ", ".join(f"{name} {round(probability * PERCENT)}%" for name, probability in shown)


@dataclass(slots=True, frozen=True)
class AdvisorDecision:
    action: int
    behavior_probability: float
    value: float
    inputs: np.ndarray
