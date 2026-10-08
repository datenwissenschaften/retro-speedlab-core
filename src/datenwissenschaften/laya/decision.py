from dataclasses import dataclass

import torch


@dataclass(slots=True, frozen=True)
class Decision:
    action: int
    probabilities: dict[str, float]
    behavior_probability: float
    value: float
    options: torch.Tensor
    pooled: torch.Tensor
    advice: int | None
