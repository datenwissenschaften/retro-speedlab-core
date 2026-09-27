import laya
import torch
from torch import nn

from datenwissenschaften.laya.precision import autocast_dtype
from datenwissenschaften.laya.question import LayaQuestion


class LayaNetwork(nn.Module):
    def __init__(self, checkpoint: str, options: dict[str, str], device: str) -> None:
        super().__init__()
        agent = laya.load(checkpoint, device="cpu")
        self.checkpoint = checkpoint
        self.question = LayaQuestion(agent.tok, agent.cfg, options)
        self.decision = agent.model
        self.decision.encoder.gradient_checkpointing_enable()
        self.decision.head_checkpointing = True
        self.to(device)
        self.dtype = autocast_dtype(self.device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, states: list[str], questions: list[str]) -> torch.Tensor:
        batch = self.question.encode(states, questions, self.device)
        with torch.autocast(self.device.type, dtype=self.dtype):
            logits, _ = self.decision(**batch)
        return logits.float()

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.decision.encoder.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        encoder = {id(parameter) for parameter in self.encoder_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in encoder]
