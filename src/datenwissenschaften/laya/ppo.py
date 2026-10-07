from typing import Any

import torch

from datenwissenschaften.laya.heads import Heads
from datenwissenschaften.laya.imitation import imitation_loss
from datenwissenschaften.laya.rollout import Rollout

GAMMA = 0.99
LAMBDA = 0.95
EPOCHS = 4
MINIBATCH = 64
CLIP = 0.2
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
LEARNING_RATE = 1e-3
MAX_GRADIENT_NORM = 0.5
NORMALIZATION_EPSILON = 1e-8

Demonstration = tuple[torch.Tensor, torch.Tensor] | None


class PpoLearner:
    def __init__(self, heads: Heads) -> None:
        self.heads = heads
        self.optimizer = torch.optim.AdamW(heads.parameters(), lr=LEARNING_RATE)

    def update(self, rollout: Rollout, demonstration: Demonstration) -> dict[str, float]:
        device = next(self.heads.parameters()).device
        options, pooled = rollout.features(device)
        actions, behavior = rollout.actions(device), rollout.behavior(device).log()
        advantages, returns = (tensor.to(device) for tensor in rollout.advantages(GAMMA, LAMBDA))
        advantages = (advantages - advantages.mean()) / (advantages.std() + NORMALIZATION_EPSILON)
        totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
        }
        imitation_total, batches = 0.0, 0
        self.heads.train()
        for _ in range(EPOCHS):
            for index in torch.randperm(len(rollout), device=device).split(MINIBATCH):
                log_probs = torch.log_softmax(self.heads.policy(options[index]), -1)
                chosen = log_probs.gather(1, actions[index, None]).squeeze(1)
                ratio = (chosen - behavior[index]).exp()
                clipped = ratio.clamp(1 - CLIP, 1 + CLIP)
                policy_loss = -torch.min(ratio * advantages[index], clipped * advantages[index]).mean()
                value_loss = (self.heads.value(pooled[index]) - returns[index]).pow(2).mean()
                entropy = -(log_probs.exp() * log_probs).sum(-1).mean()
                loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * entropy
                if demonstration is not None:
                    imitation = imitation_loss(self.heads.policy, *demonstration)
                    loss = loss + imitation
                    imitation_total += imitation.item()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.heads.parameters(), MAX_GRADIENT_NORM)
                self.optimizer.step()
                for name, value in (
                    ("policy_loss", policy_loss),
                    ("value_loss", value_loss),
                    ("entropy", entropy),
                    ("approx_kl", (behavior[index] - chosen).mean()),
                    ("clip_fraction", ((ratio - 1).abs() > CLIP).float().mean()),
                ):
                    totals[name] += value.item()
                batches += 1
        self.heads.eval()
        metrics = {name: total / batches for name, total in totals.items()}
        metrics["explained_variance"] = explained_variance(returns, rollout)
        metrics["imitation_loss"] = imitation_total / batches
        metrics["demonstration_decisions"] = 0 if demonstration is None else len(demonstration[1])
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])


def explained_variance(returns: torch.Tensor, rollout: Rollout) -> float:
    values = torch.tensor([decision.value for decision in rollout.decisions], device=returns.device)
    variance = returns.var()
    if variance.item() == 0:
        return 0.0
    return float(1 - (returns - values).var() / variance)
