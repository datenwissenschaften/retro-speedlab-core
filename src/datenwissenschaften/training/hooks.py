from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from datenwissenschaften.environment.wrapper import Observation
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.training.episode_record import EpisodeRecord


@dataclass(slots=True, frozen=True)
class Transition:
    timesteps: int
    observation: Observation
    decision: Decision
    frames: list[np.ndarray]
    reward: float
    done: bool
    info: dict[str, Any]


class TrainingHook(Protocol):
    def on_step(self, transition: Transition) -> None: ...

    def on_episode_end(self, episode: EpisodeRecord) -> None: ...

    def on_update(self) -> None: ...
