from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from datenwissenschaften.rollout import Rollout, Step

GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 4
CLIP = 0.2
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
MAX_GRADIENT_NORM = 0.5
NORMALIZATION_EPSILON = 1e-8

Evaluate = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
Imitate = Callable[[torch.Tensor], torch.Tensor]


class PpoLearner:
    def __init__(self, parts: tuple[nn.Module, ...], learning_rate: float, minibatch: int) -> None:
        self.parts = parts
        self.minibatch = minibatch
        self.optimizer = torch.optim.AdamW([p for part in parts for p in part.parameters()], lr=learning_rate)

    def update(self, rollout: Rollout[Step], evaluate: Evaluate, imitate: Imitate) -> dict[str, float]:
        device = next(self.parts[0].parameters()).device
        actions, behavior = rollout.actions(device), rollout.behavior(device).log()
        advantages, returns = (tensor.to(device) for tensor in rollout.advantages(GAMMA, LAMBDA))
        advantages = (advantages - advantages.mean()) / (advantages.std() + NORMALIZATION_EPSILON)
        names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "imitation_loss")
        totals = dict.fromkeys(names, 0.0)
        batches = 0
        for part in self.parts:
            part.train()
        for _ in range(EPOCHS):
            for index in torch.randperm(len(rollout), device=device).split(self.minibatch):
                log_probs, values = evaluate(index)
                chosen = log_probs.gather(1, actions[index, None]).squeeze(1)
                ratio = (chosen - behavior[index]).exp()
                clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
                policy_loss = -torch.min(ratio * advantages[index], clipped * advantages[index]).mean()
                value_loss = (values - returns[index]).pow(2).mean()
                entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
                imitation = imitate(index)
                loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy + imitation
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for part in self.parts:
                    torch.nn.utils.clip_grad_norm_(part.parameters(), MAX_GRADIENT_NORM)
                self.optimizer.step()
                for name, value in zip(
                    names,
                    (
                        policy_loss,
                        value_loss,
                        entropy,
                        (behavior[index] - chosen).mean(),
                        ((ratio - 1).abs() > CLIP).float().mean(),
                        imitation,
                    ),
                    strict=True,
                ):
                    totals[name] += value.item()
                batches += 1
        for part in self.parts:
            part.eval()
        metrics = {name: total / batches for name, total in totals.items()}
        metrics["explained_variance"] = explained_variance(returns, rollout.values().to(device))
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    variance = returns.var()
    if variance.item() == 0:
        return 0.0
    return float(1 - (returns - values).var() / variance)
