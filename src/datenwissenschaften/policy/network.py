import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datenwissenschaften.tas.paths import atomic_write

HIDDEN = 256


class PolicyNet(nn.Module):
    def __init__(self, inputs: int, actions: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(inputs, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh())
        self.policy = nn.Linear(HIDDEN, actions)
        self.value = nn.Linear(HIDDEN, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observations)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def export_policy(net: PolicyNet, path: Path) -> None:
    layers = (net.body[0], net.body[2], net.policy)
    arrays = {}
    for index, layer in enumerate(layers):
        arrays[f"weight{index}"] = layer.weight.detach().cpu().numpy()
        arrays[f"bias{index}"] = layer.bias.detach().cpu().numpy()
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    atomic_write(path, buffer.getvalue())
