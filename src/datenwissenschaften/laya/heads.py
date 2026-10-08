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
    def __init__(self, width: int, option_count: int) -> None:
        super().__init__()
        inputs = width * (option_count + 1)
        self.net = nn.Sequential(
            nn.LayerNorm(inputs), nn.Linear(inputs, VALUE_HIDDEN), nn.GELU(), nn.Linear(VALUE_HIDDEN, 1)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, options: torch.Tensor, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([pooled, options.flatten(1)], -1)).squeeze(-1)


class Heads(nn.Module):
    def __init__(self, scorer: nn.Module, width: int, option_count: int) -> None:
        super().__init__()
        self.policy = PolicyHead(scorer)
        self.value = ValueHead(width, option_count)
