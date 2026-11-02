from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical

from datenwissenschaften.policy.network import PolicyNet
from datenwissenschaften.policy.prior import observation
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import leaves

ROLLOUT_DECISIONS = 2048
EPISODE_DECISIONS = 128
EPOCHS = 4
MINIBATCH = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
NOVELTY_REWARD = 0.01
EXIT_REWARD = 10.0
DEATH_PENALTY = 1.0
STEADY = 1e-8


@dataclass(slots=True)
class Rollout:
    observations: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]
    exits: int


def reward_of(outcome_reward: float, novelty: int, exited: bool, alive: bool) -> float:
    return outcome_reward + NOVELTY_REWARD * novelty + EXIT_REWARD * exited - DEATH_PENALTY * (not alive)


def collect(runner: SegmentRunner, start: Start, net: PolicyNet, device: torch.device, novelty: RamNovelty) -> Rollout:
    rollout = Rollout([], [], [], [], [], [], 0)
    runner.restore(start)
    decisions = 0
    while len(rollout.actions) < ROLLOUT_DECISIONS:
        seen = observation(runner.memory())
        with torch.no_grad():
            logits, value = net(torch.from_numpy(seen).to(device)[None])
        distribution = Categorical(logits=logits[0])
        action = distribution.sample()
        outcome = runner.decide(int(action))
        decisions += 1
        exited = leaves(runner, start, outcome.state, outcome.won)
        ended = exited or not outcome.alive or outcome.state != start.state or decisions >= EPISODE_DECISIONS
        rollout.observations.append(seen)
        rollout.actions.append(int(action))
        rollout.log_probs.append(float(distribution.log_prob(action)))
        rollout.values.append(float(value[0]))
        rollout.rewards.append(reward_of(outcome.reward, novelty.see(runner.memory()), exited, outcome.alive))
        rollout.dones.append(ended)
        rollout.exits += exited
        if ended:
            runner.restore(start)
            decisions = 0
    return rollout


def advantages(rollout: Rollout) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray([*rollout.values, 0.0], dtype=np.float32)
    gains = np.zeros(len(rollout.rewards), dtype=np.float32)
    running = 0.0
    for index in reversed(range(len(rollout.rewards))):
        going = 0.0 if rollout.dones[index] else 1.0
        delta = rollout.rewards[index] + GAMMA * values[index + 1] * going - values[index]
        running = delta + GAMMA * LAMBDA * going * running
        gains[index] = running
    return gains, gains + values[:-1]


def update(net: PolicyNet, optimizer: torch.optim.Optimizer, rollout: Rollout, device: torch.device) -> float:
    gains, returns = advantages(rollout)
    observations = torch.from_numpy(np.stack(rollout.observations)).to(device)
    actions = torch.as_tensor(rollout.actions, device=device)
    old_log_probs = torch.as_tensor(rollout.log_probs, device=device)
    gains_t = torch.from_numpy((gains - gains.mean()) / (gains.std() + STEADY)).to(device)
    returns_t = torch.from_numpy(returns).to(device)
    loss = torch.zeros((), device=device)
    for _ in range(EPOCHS):
        for batch in torch.randperm(len(actions), device=device).split(MINIBATCH):
            logits, values = net(observations[batch])
            distribution = Categorical(logits=logits)
            ratio = torch.exp(distribution.log_prob(actions[batch]) - old_log_probs[batch])
            clipped = torch.clamp(ratio, 1 - CLIP, 1 + CLIP)
            policy_loss = -torch.min(ratio * gains_t[batch], clipped * gains_t[batch]).mean()
            value_loss = (returns_t[batch] - values).pow(2).mean()
            loss = policy_loss + VALUE_WEIGHT * value_loss - ENTROPY_WEIGHT * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()
