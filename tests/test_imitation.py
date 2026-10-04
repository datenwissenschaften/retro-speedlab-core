import torch
from torch import nn

from datenwissenschaften.laya.imitation import DemonstrationStep, imitate

LEARNING_RATE = 0.5


class BiasNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(2))

    @property
    def device(self) -> torch.device:
        return self.bias.device

    def forward(self, states: list[str], questions: list[str]) -> torch.Tensor:
        return self.bias.expand(len(states), 2)


def test_imitation_moves_the_policy_toward_the_demonstrated_move():
    network = BiasNetwork()
    optimizer = torch.optim.SGD(network.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cpu", enabled=False)

    loss = imitate(network, scaler, [DemonstrationStep("{}", "Which move?", 1)] * 3, 2)
    optimizer.step()

    assert loss == torch.log(torch.tensor(2.0)).item()
    assert torch.softmax(network.bias, -1)[1] > 0.5


def test_no_demonstrations_leave_the_gradients_alone():
    network = BiasNetwork()

    assert imitate(network, torch.amp.GradScaler("cpu", enabled=False), [], 2) == 0.0
    assert network.bias.grad is None
