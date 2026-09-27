import bitsandbytes as bnb
import torch

from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.laya.trust_region import TrustRegion

ENCODER_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GAMMA = 0.99
PROBABILITY_FLOOR = 1e-8
MAX_GRADIENT_NORM = 1.0
MINIBATCH_SIZE = 8


class GroupRelativeLearner:
    def __init__(self, network: LayaNetwork) -> None:
        self.network = network
        self.base_learning_rates = (ENCODER_LEARNING_RATE, HEAD_LEARNING_RATE)
        self.optimizer = bnb.optim.AdamW8bit(
            [
                {"params": network.encoder_parameters(), "lr": ENCODER_LEARNING_RATE},
                {"params": network.head_parameters(), "lr": HEAD_LEARNING_RATE},
            ],
            weight_decay=WEIGHT_DECAY,
        )
        self.trust_region = TrustRegion()

    def update(self, rollout: Rollout) -> dict[str, float]:
        previous = torch.as_tensor(rollout.probabilities, device=self.network.device).clamp_min(PROBABILITY_FLOOR)
        policy_loss, entropy = self._step(rollout, previous)
        kl = self._measure_kl(rollout, previous)
        self.trust_region.adapt(kl, entropy, previous.shape[1])
        for group, base in zip(self.optimizer.param_groups, self.base_learning_rates, strict=True):
            group["lr"] = base * self.trust_region.learning_rate_scale
        return {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "kl": kl,
            "entropy_coefficient": self.trust_region.entropy_coefficient,
            "learning_rate_scale": self.trust_region.learning_rate_scale,
        }

    def _step(self, rollout: Rollout, previous: torch.Tensor) -> tuple[float, float]:
        actions = torch.as_tensor(rollout.actions, device=self.network.device)
        sampled = torch.as_tensor(rollout.behavior_probabilities, device=self.network.device)
        importance = previous.gather(1, actions[:, None]).squeeze(1) / sampled
        weighted = rollout.group_relative_advantages(GAMMA).to(self.network.device) * importance
        count = len(rollout)
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss_total, entropy_total = 0.0, 0.0
        for start in range(0, count, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            log_probs = torch.log_softmax(self.network(rollout.states[start:end], rollout.questions[start:end]), -1)
            chosen = log_probs.gather(1, actions[start:end, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            policy_loss = -(weighted[start:end] * chosen).sum()
            ((policy_loss - self.trust_region.entropy_coefficient * entropy.sum()) / count).backward()
            policy_loss_total += policy_loss.item()
            entropy_total += entropy.sum().item()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()
        self.network.eval()
        return policy_loss_total / count, entropy_total / count

    @torch.no_grad()
    def _measure_kl(self, rollout: Rollout, previous: torch.Tensor) -> float:
        log_probs = torch.log_softmax(
            self.network(rollout.states[:MINIBATCH_SIZE], rollout.questions[:MINIBATCH_SIZE]), -1
        )
        probe = previous[:MINIBATCH_SIZE]
        return float((probe * (probe.log() - log_probs)).sum(-1).mean())
