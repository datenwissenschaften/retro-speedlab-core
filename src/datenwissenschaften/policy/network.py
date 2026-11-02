import io
from pathlib import Path

import numpy as np
import torch
from torch import nn

from datenwissenschaften.policy.prior import BIAS, VALUE_BIAS, VALUE_WEIGHT, WEIGHT, Weights
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


def numpy_weights(net: PolicyNet) -> Weights:
    arrays = {VALUE_WEIGHT: tensor(net.value.weight)[0], VALUE_BIAS: tensor(net.value.bias)[0]}
    for index, layer in enumerate((net.body[0], net.body[2], net.policy)):
        arrays[f"{WEIGHT}{index}"] = tensor(layer.weight)
        arrays[f"{BIAS}{index}"] = tensor(layer.bias)
    return arrays


def tensor(parameter: torch.Tensor) -> np.ndarray:
    return parameter.detach().cpu().numpy()


def export_policy(net: PolicyNet, path: Path) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, **numpy_weights(net))
    atomic_write(path, buffer.getvalue())
