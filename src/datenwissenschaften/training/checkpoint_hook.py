from loguru import logger

from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.telemetry import publish_metadata

DISPLAY_NAME = "Laya"
DESCRIPTION = "One Laya model per state chooses every action; group-relative policy gradients fine-tune all weights."


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
        logger.debug(f"{self.models.active} model saved at {self.models.agent.num_timesteps:,} trained decisions")
