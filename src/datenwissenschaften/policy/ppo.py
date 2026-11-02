import numpy as np
import torch
from torch.distributions import Categorical

from datenwissenschaften.policy.network import PolicyNet
from datenwissenschaften.policy.rollout import Rollout

EPOCHS = 4
MINIBATCH = 256
GAMMA = 0.99
LAMBDA = 0.95
CLIP = 0.2
VALUE_LOSS_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
STEADY = 1e-8


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
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss - ENTROPY_WEIGHT * distribution.entropy().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return loss.item()
