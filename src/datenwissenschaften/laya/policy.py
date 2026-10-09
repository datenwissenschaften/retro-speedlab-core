import io
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from datenwissenschaften.laya.heads import Heads
from datenwissenschaften.ppo import PpoLearner


@dataclass(slots=True)
class Policy:
    heads: Heads
    learner: PpoLearner
    num_timesteps: int
    last_update: dict[str, float]

    def checkpoint(self) -> io.BytesIO:
        buffer = io.BytesIO()
        torch.save(
            {
                "heads": self.heads.state_dict(),
                "learner": self.learner.state_dict(),
                "num_timesteps": self.num_timesteps,
                "last_update": self.last_update,
            },
            buffer,
        )
        return buffer

    def restore(self, checkpoint: dict[str, Any]) -> None:
        heads = checkpoint["heads"]
        broken = {name.split(".")[0] for name, tensor in heads.items() if not tensor.isfinite().all()}
        for part in sorted(broken):
            logger.warning(
                f"Laya's {part} head in this checkpoint is not finite; it starts again from Laya's own judgement"
            )
        fresh = self.heads.state_dict()
        self.heads.load_state_dict(
            {name: fresh[name] if name.split(".")[0] in broken else tensor for name, tensor in heads.items()}
        )
        if not broken:
            self.learner.load_state_dict(checkpoint["learner"])
        self.num_timesteps = int(checkpoint["num_timesteps"])
        self.last_update = checkpoint["last_update"]
