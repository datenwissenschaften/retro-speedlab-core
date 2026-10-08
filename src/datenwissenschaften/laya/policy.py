import io
from dataclasses import dataclass
from typing import Any

import torch

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
        self.heads.load_state_dict(checkpoint["heads"])
        self.learner.load_state_dict(checkpoint["learner"])
        self.num_timesteps = int(checkpoint["num_timesteps"])
        self.last_update = checkpoint["last_update"]
