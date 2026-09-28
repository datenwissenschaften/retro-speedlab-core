from typing import Any

import bitsandbytes as bnb
import torch

from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.precision import minibatch_size
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.laya.trust_region import TrustRegion
from datenwissenschaften.laya.weight_snapshot import WeightSnapshot

ENCODER_LEARNING_RATE = 1e-9
HEAD_LEARNING_RATE = 1e-8
WEIGHT_DECAY = 0.01
GAMMA = 0.99
PROBABILITY_FLOOR = 1e-8
MAX_GRADIENT_NORM = 1.0
MAX_BACKTRACKS = 3
ENTROPY_COEFFICIENT = 0.001


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
        self.minibatch_size = minibatch_size(network.device)
        self.scaler = torch.amp.GradScaler(network.device.type, enabled=network.dtype == torch.float16)
        self.snapshot = WeightSnapshot(list(network.parameters()))

    def update(self, rollout: Rollout) -> dict[str, float]:
        previous = torch.as_tensor(rollout.probabilities, device=self.network.device).clamp_min(PROBABILITY_FLOOR)
        self.snapshot.capture()
        policy_loss, entropy = self._step(rollout, previous)
        step_kl = self._measure_kl(rollout, previous)
        kl = self._backtrack(rollout, previous, step_kl)
        self.trust_region.adapt(step_kl)
        for group, base in zip(self.optimizer.param_groups, self.base_learning_rates, strict=True):
            group["lr"] = base * self.trust_region.learning_rate_scale
        return {
            "policy_loss": policy_loss,
            "entropy": entropy,
            "step_kl": step_kl,
            "kl": kl,
            "learning_rate_scale": self.trust_region.learning_rate_scale,
        }

    def _backtrack(self, rollout: Rollout, previous: torch.Tensor, kl: float) -> float:
        for _ in range(MAX_BACKTRACKS):
            if self.trust_region.accepts(kl):
                return kl
            self.snapshot.blend(self.trust_region.backtrack_fraction(kl))
            kl = self._measure_kl(rollout, previous)
        if self.trust_region.accepts(kl):
            return kl
        self.snapshot.blend(0.0)
        return self._measure_kl(rollout, previous)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self.optimizer.state_dict(),
            "trust_region": self.trust_region.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.trust_region.load_state_dict(state["trust_region"])
        self.scaler.load_state_dict(state["scaler"])

    def _step(self, rollout: Rollout, previous: torch.Tensor) -> tuple[float, float]:
        actions = torch.as_tensor(rollout.actions, device=self.network.device)
        sampled = torch.as_tensor(rollout.behavior_probabilities, device=self.network.device)
        importance = previous.gather(1, actions[:, None]).squeeze(1) / sampled
        weighted = rollout.group_relative_advantages(GAMMA).to(self.network.device) * importance
        count = len(rollout)
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        policy_loss_total, entropy_total = 0.0, 0.0
        for start in range(0, count, self.minibatch_size):
            end = start + self.minibatch_size
            log_probs = torch.log_softmax(self.network(rollout.states[start:end], rollout.questions[start:end]), -1)
            chosen = log_probs.gather(1, actions[start:end, None]).squeeze(1)
            entropy = -(log_probs.exp() * log_probs).sum(-1)
            policy_loss = -(weighted[start:end] * chosen).sum()
            loss = (policy_loss - ENTROPY_COEFFICIENT * entropy.sum()) / count
            self.scaler.scale(loss).backward()
            policy_loss_total += policy_loss.item()
            entropy_total += entropy.sum().item()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRADIENT_NORM)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.network.eval()
        return policy_loss_total / count, entropy_total / count

    @torch.no_grad()
    def _measure_kl(self, rollout: Rollout, previous: torch.Tensor) -> float:
        log_probs = torch.log_softmax(
            self.network(rollout.states[: self.minibatch_size], rollout.questions[: self.minibatch_size]), -1
        )
        probe = previous[: self.minibatch_size]
        return float((probe * (probe.log() - log_probs)).sum(-1).mean())
