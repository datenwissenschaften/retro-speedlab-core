from typing import Generic, Protocol, TypeVar

import torch


class Step(Protocol):
    @property
    def action(self) -> int: ...

    @property
    def behavior_probability(self) -> float: ...

    @property
    def value(self) -> float: ...


S = TypeVar("S", bound=Step)


class Rollout(Generic[S]):
    def __init__(self) -> None:
        self.decisions: list[S] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.cuts: list[bool] = []

    def __len__(self) -> int:
        return len(self.decisions)

    @classmethod
    def joined(cls, parts: list["Rollout[S]"]) -> "Rollout[S]":
        rollout: Rollout[S] = cls()
        for part in parts:
            rollout.decisions += part.decisions
            rollout.rewards += part.rewards
            rollout.dones += part.dones
            rollout.cuts += [*part.cuts[:-1], True]
        return rollout

    def add(self, decision: S, reward: float, terminal: bool, truncated: bool) -> None:
        self.decisions.append(decision)
        self.rewards.append(reward)
        self.dones.append(terminal)
        self.cuts.append(truncated)

    def advantages(self, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        values = [decision.value for decision in self.decisions]
        advantages = torch.zeros(len(self))
        running = 0.0
        for step in reversed(range(len(self))):
            last = self.cuts[step] or step + 1 == len(self)
            following = values[step] if last else values[step + 1]
            following = 0.0 if self.dones[step] else following
            delta = self.rewards[step] + gamma * following - values[step]
            running = delta + gamma * lam * running * (not (self.dones[step] or last))
            advantages[step] = running
        returns = advantages + torch.tensor(values)
        return advantages, returns

    def values(self) -> torch.Tensor:
        return torch.tensor([decision.value for decision in self.decisions])

    def actions(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([decision.action for decision in self.decisions], device=device)

    def behavior(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([decision.behavior_probability for decision in self.decisions], device=device)
