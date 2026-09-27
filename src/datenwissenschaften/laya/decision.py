from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Decision:
    action: int
    probabilities: dict[str, float]
    behavior_probability: float
