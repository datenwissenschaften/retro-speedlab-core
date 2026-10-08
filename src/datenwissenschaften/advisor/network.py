import torch
from torch import nn

HIDDEN = 256
INPUT_CLIP = 5.0
VARIANCE_EPSILON = 1e-8
POLICY_GAIN = 0.01


class Normalizer(nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0))

    def update(self, batch: torch.Tensor) -> None:
        count = self.count + len(batch)
        delta = batch.mean(0) - self.mean
        spread = self.variance * self.count + batch.var(0, unbiased=False) * len(batch)
        self.variance.copy_((spread + delta.pow(2) * self.count * len(batch) / count) / count)
        self.mean.add_(delta * len(batch) / count)
        self.count.copy_(count)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        scaled = (inputs - self.mean) / (self.variance + VARIANCE_EPSILON).sqrt()
        return scaled.clamp(-INPUT_CLIP, INPUT_CLIP)


def _trunk(inputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())


class ActorCritic(nn.Module):
    def __init__(self, inputs: int, actions: int) -> None:
        super().__init__()
        self.normalizer = Normalizer(inputs)
        self.policy = nn.Sequential(_trunk(inputs), nn.Linear(HIDDEN, actions))
        self.value = nn.Sequential(_trunk(inputs), nn.Linear(HIDDEN, 1))
        nn.init.orthogonal_(self.policy[-1].weight, POLICY_GAIN)
        nn.init.zeros_(self.policy[-1].bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.normalizer(inputs)
        return torch.log_softmax(self.policy(normalized), -1), self.value(normalized).squeeze(-1)
