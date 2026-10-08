import math

from loguru import logger

from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.stall_watch import StallWatch
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.telemetry import publish_metadata

DISPLAY_NAME = "Laya"
DESCRIPTION = (
    "Laya reads the game and the advice of a fast practice coach, then decides every move; "
    "small heads per state learn with PPO and imitation."
)


def learning_metadata(models: StateModels) -> dict[str, float]:
    update = models.agent.last_update
    uniform = math.log(len(models.agent.network.question.options))
    learning = {
        "num_timesteps": models.agent.num_timesteps,
        "entropy_share": round(update["entropy"] / uniform, 3),
        "explained_variance": round(update["explained_variance"], 3),
    }
    if "advisor_agreement" in update:
        learning["advisor_agreement"] = round(update["advisor_agreement"], 3)
    return learning


def model_metadata(models: StateModels) -> dict[str, object]:
    return {
        "display_name": DISPLAY_NAME,
        "description": DESCRIPTION,
        "laya": {"state": models.active, **models.agent.metadata()},
    }


class CheckpointHook:
    def __init__(self, models: StateModels) -> None:
        self.models = models
        self.stall_watch = StallWatch()

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        self.models.save()
        publish_metadata("model", model_metadata(self.models), replace=True)
        learning = learning_metadata(self.models)
        state = self.models.require_active()
        learning["stalled"] = self.stall_watch.observe(
            state, self.models.agent.num_timesteps, learning["entropy_share"], learning["explained_variance"]
        )
        publish_metadata("state_models", {state: learning})
        logger.debug(
            f"{self.models.active} model queued for saving at {self.models.agent.num_timesteps:,} trained decisions"
        )
