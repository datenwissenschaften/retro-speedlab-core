import torch

from datenwissenschaften.laya.decision import Decision


class Rollout:
    def __init__(self) -> None:
        self.decisions: list[Decision] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def __len__(self) -> int:
        return len(self.decisions)

    def add(self, decision: Decision, reward: float, done: bool) -> None:
        self.decisions.append(decision)
        self.rewards.append(reward)
        self.dones.append(done)

    def advantages(self, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
        values = [decision.value for decision in self.decisions]
        advantages = torch.zeros(len(self))
        running = 0.0
        for step in reversed(range(len(self))):
            following = values[step + 1] if step + 1 < len(self) else values[step]
            following = 0.0 if self.dones[step] else following
            delta = self.rewards[step] + gamma * following - values[step]
            running = delta + gamma * lam * running * (not self.dones[step])
            advantages[step] = running
        returns = advantages + torch.tensor(values)
        return advantages, returns

    def features(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        options = torch.stack([decision.options for decision in self.decisions]).to(device)
        pooled = torch.stack([decision.pooled for decision in self.decisions]).to(device)
        return options, pooled

    def actions(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([decision.action for decision in self.decisions], device=device)

    def behavior(self, device: torch.device) -> torch.Tensor:
        return torch.tensor([decision.behavior_probability for decision in self.decisions], device=device)
