import os

import wandb
from loguru import logger

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.paths import tas_dir

API_KEY = "WANDB_API_KEY"
PROJECT = "retro-speedlab"
ONLINE = "online"
DISABLED = "disabled"


def start_experiment(settings: RetroSpeedlabConfig, job: str) -> wandb.Run:
    tracked = API_KEY in os.environ
    if not tracked:
        logger.warning(f"The {job} is not tracked in Weights & Biases: {API_KEY} is not set")
    speedrun = settings.speedrun
    return wandb.init(
        project=PROJECT,
        group=settings.ui.release,
        job_type=job,
        name=f"{job} {settings.ui.release}",
        tags=[settings.training.game, speedrun.game.category],
        config={
            "game": settings.training.game,
            "category": speedrun.game.category,
            "scouts": speedrun.scouts,
            "backtrackers": speedrun.backtrackers,
            "beam_scouts": speedrun.beam_scouts,
            "vision_scouts": speedrun.vision_scouts,
            "polishers": speedrun.polishers,
            "segment_seconds": speedrun.segment_seconds,
            "policy_emulators": speedrun.policy_emulators,
        },
        dir=tas_dir(settings),
        mode=ONLINE if tracked else DISABLED,
    )
