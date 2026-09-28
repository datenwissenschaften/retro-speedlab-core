import torch
from torch import nn


class WeightSnapshot:
    def __init__(self, parameters: list[nn.Parameter]) -> None:
        self.parameters = parameters
        self.saved = [torch.empty_like(parameter, device="cpu") for parameter in parameters]

    @torch.no_grad()
    def capture(self) -> None:
        for saved, parameter in zip(self.saved, self.parameters, strict=True):
            saved.copy_(parameter)

    @torch.no_grad()
    def blend(self, fraction: float) -> None:
        for saved, parameter in zip(self.saved, self.parameters, strict=True):
            before = saved.to(parameter.device)
            parameter.sub_(before).mul_(fraction).add_(before)
