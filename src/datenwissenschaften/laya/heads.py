import copy

import torch
from torch import nn

VALUE_HIDDEN = 256


class PolicyHead(nn.Module):
    def __init__(self, scorer: nn.Module) -> None:
        super().__init__()
        self.scorer = copy.deepcopy(scorer).float().requires_grad_(True)

    def forward(self, options: torch.Tensor) -> torch.Tensor:
        return self.scorer(options).squeeze(-1)


class ValueHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, VALUE_HIDDEN), nn.GELU(), nn.Linear(VALUE_HIDDEN, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled).squeeze(-1)


class Heads(nn.Module):
    def __init__(self, scorer: nn.Module, width: int) -> None:
        super().__init__()
        self.policy = PolicyHead(scorer)
        self.value = ValueHead(width)
