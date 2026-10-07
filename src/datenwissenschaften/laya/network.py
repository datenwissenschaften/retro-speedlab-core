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
        self.to(device)
        self.requires_grad_(False)
        self.eval()
        self.dtype = autocast_dtype(self.device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def width(self) -> int:
        return self.decision.type_emb.embedding_dim

    @property
    def scorer(self) -> nn.Module:
        return self.decision.scorer

    @torch.no_grad()
    def features(self, states: list[str], questions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        batch, orders = self.question.encode(states, questions, self.device)
        model = self.decision
        with torch.autocast(self.device.type, dtype=self.dtype):
            hidden = model.encoder(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).last_hidden_state
            hidden = hidden + model.type_emb(batch["qtype"])[:, None, :]
            if model.head is not None:
                padding = ~batch["attention_mask"].bool()
                for layer in model.head.layers:
                    hidden = layer(hidden, src_key_padding_mask=padding)
        hidden = hidden.float()
        markers = torch.gather(hidden, 1, batch["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1)))
        options = torch.empty_like(markers).scatter_(1, orders[:, :, None].expand(-1, -1, hidden.size(-1)), markers)
        return options, hidden[:, 0]
