import random
from dataclasses import dataclass

import torch

from datenwissenschaften.laya.network import LayaNetwork

IMITATION_WEIGHT = 1.0


@dataclass(slots=True, frozen=True)
class DemonstrationStep:
    state: str
    question: str
    action: int


def imitate(
    network: LayaNetwork, scaler: torch.amp.GradScaler, demonstrations: list[DemonstrationStep], batch_size: int
) -> float:
    if not demonstrations:
        return 0.0
    sample = random.sample(demonstrations, min(len(demonstrations), batch_size))
    actions = torch.as_tensor([step.action for step in sample], device=network.device)
    log_probs = torch.log_softmax(network([step.state for step in sample], [step.question for step in sample]), -1)
    loss = -log_probs.gather(1, actions[:, None]).mean()
    scaler.scale(IMITATION_WEIGHT * loss).backward()
    return loss.item()
