import math

from loguru import logger

from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.telemetry import publish_metadata

DISPLAY_NAME = "Laya"
DESCRIPTION = "Laya reads the game frozen; small policy and value heads per state learn with PPO."


def learning_metadata(models: StateModels) -> dict[str, object]:
    update = models.agent.last_update
    uniform = math.log(len(models.agent.network.question.options))
    return {
        "num_timesteps": models.agent.num_timesteps,
        "entropy_share": round(update["entropy"] / uniform, 3),
        "explained_variance": round(update["explained_variance"], 3),
    }


def model_metadata(models: StateModels) -> dict[str, object]:
    return {
        "display_name": DISPLAY_NAME,
        "description": DESCRIPTION,
        "laya": {"state": models.active, **models.agent.metadata()},
    }


class CheckpointHook:
    def __init__(self, models: StateModels) -> None:
        self.models = models

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        self.models.save()
        publish_metadata("model", model_metadata(self.models), replace=True)
        publish_metadata("state_models", {self.models.require_active(): learning_metadata(self.models)})
        logger.debug(
            f"{self.models.active} model queued for saving at {self.models.agent.num_timesteps:,} trained decisions"
        )
