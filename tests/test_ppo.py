from dataclasses import dataclass

import torch
from torch import nn

from datenwissenschaften.ppo import PpoLearner
from datenwissenschaften.rollout import Rollout

ACTIONS = 7
EXPLORATION = 0.2
DECISIONS = 512


@dataclass(slots=True, frozen=True)
class Move:
    action: int
    behavior_probability: float
    policy_probability: float
    value: float


def confident_rollout(policy: torch.Tensor) -> Rollout[Move]:
    behavior = (1 - EXPLORATION) * policy + EXPLORATION / ACTIONS
    generator = torch.Generator().manual_seed(0)
    rollout: Rollout[Move] = Rollout()
    for action in torch.multinomial(behavior, DECISIONS, replacement=True, generator=generator).tolist():
        move = Move(action, float(behavior[action]), float(policy[action]), 0.0)
        rollout.add(move, float(torch.rand(1, generator=generator)), False, False)
    return rollout


def test_exploration_does_not_count_as_a_policy_step():
    logits = nn.Parameter(torch.tensor([5.0] + [0.0] * (ACTIONS - 1)))
    value = nn.Parameter(torch.zeros(()))
    holder, critic = nn.Module(), nn.Module()
    holder.logits, critic.value = logits, value
    rollout = confident_rollout(torch.softmax(logits.detach(), -1))

    def evaluate(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.log_softmax(logits, -1).expand(len(index), -1), value.expand(len(index))

    metrics = PpoLearner((holder, critic), 1e-4, 64).update(rollout, evaluate, lambda index: value.new_zeros(()))

    assert metrics["clip_fraction"] < 0.05
    assert abs(metrics["approx_kl"]) < 0.01


def test_an_explored_move_the_policy_had_ruled_out_does_not_poison_the_policy():
    logits = nn.Parameter(torch.tensor([200.0] + [0.0] * (ACTIONS - 1)))
    value = nn.Parameter(torch.zeros(()))
    holder, critic = nn.Module(), nn.Module()
    holder.logits, critic.value = logits, value
    rollout: Rollout[Move] = Rollout()
    for step in range(DECISIONS):
        action = step % ACTIONS
        policy = float(torch.softmax(logits.detach(), -1)[action])
        rollout.add(
            Move(action, (1 - EXPLORATION) * policy + EXPLORATION / ACTIONS, policy, 0.0), -float(action), False, False
        )

    def evaluate(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.log_softmax(logits, -1).expand(len(index), -1), value.expand(len(index))

    metrics = PpoLearner((holder, critic), 1e-4, 64).update(rollout, evaluate, lambda index: value.new_zeros(()))

    assert torch.isfinite(logits).all()
    assert metrics["skipped_minibatches"] == 0
