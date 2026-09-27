from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.telemetry import publish_metadata

DISPLAY_NAME = "Laya"
DESCRIPTION = "Laya chooses every action; group-relative policy gradients fine-tune all of its weights."


def model_metadata(agent: LayaAgent) -> dict[str, object]:
    return {"display_name": DISPLAY_NAME, "description": DESCRIPTION, "laya": agent.metadata()}


class CheckpointHook:
    def __init__(self, context: RunContext, agent: LayaAgent) -> None:
        self.context = context
        self.agent = agent

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        self.context.model_dir.mkdir(parents=True, exist_ok=True)
        self.agent.save(self.context.model_path)
        publish_metadata("model", model_metadata(self.agent), replace=True)
        logger.debug(f"Checkpoint saved at {self.agent.num_timesteps:,} steps")
