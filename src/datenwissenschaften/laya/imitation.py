from dataclasses import dataclass

import torch

from datenwissenschaften.laya.heads import PolicyHead

IMITATION_WEIGHT = 1.0


@dataclass(slots=True, frozen=True)
class DemonstrationStep:
    state: str
    question: str
    action: int


def imitation_loss(policy: PolicyHead, options: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(policy(options), -1)
    return -IMITATION_WEIGHT * log_probs.gather(1, actions[:, None]).mean()
