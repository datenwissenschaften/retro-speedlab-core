import numpy as np
import torch

from datenwissenschaften.laya.decision import Decision

NORMALIZATION_EPSILON = 1e-6


class Rollout:
    def __init__(self) -> None:
        self.states: list[str] = []
        self.questions: list[str] = []
        self.actions: list[int] = []
        self.probabilities: list[list[float]] = []
        self.behavior_probabilities: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __len__(self) -> int:
        return len(self.actions)

    def add(self, state: str, question: str, decision: Decision, reward: float, done: bool) -> None:
        self.states.append(state)
        self.questions.append(question)
        self.actions.append(decision.action)
        self.probabilities.append(list(decision.probabilities.values()))
        self.behavior_probabilities.append(decision.behavior_probability)
        self.rewards.append(reward)
        self.dones.append(done)

    def group_relative_advantages(self, gamma: float) -> torch.Tensor:
        returns = np.zeros(len(self), dtype=np.float32)
        running = 0.0
        for step in reversed(range(len(self))):
            running = self.rewards[step] + gamma * running * (not self.dones[step])
            returns[step] = running
        advantages = torch.as_tensor(returns)
        return (advantages - advantages.mean()) / (advantages.std() + NORMALIZATION_EPSILON)
