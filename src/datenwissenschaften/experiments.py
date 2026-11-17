import os
from typing import Any

import wandb
from loguru import logger
from sentry_sdk.integrations.logging import ignore_logger

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.paths import tas_dir

API_KEY = "WANDB_API_KEY"
SDK_TELEMETRY = "WANDB_ERROR_REPORTING"
SDK_LOGGERS = "wandb*"
PROJECT = "retro-speedlab"
ONLINE = "online"
DISABLED = "disabled"
JOB = "policy"


def start_experiment(settings: RetroSpeedlabConfig, hyperparameters: dict[str, Any]) -> wandb.Run:
    tracked = API_KEY in os.environ
    os.environ[SDK_TELEMETRY] = "false"
    ignore_logger(SDK_LOGGERS)
    if not tracked:
        logger.warning(f"The {JOB} is not tracked in Weights & Biases: {API_KEY} is not set")
    speedrun = settings.speedrun
    return wandb.init(
        project=PROJECT,
        group=settings.ui.release,
        job_type=JOB,
        name=f"{JOB} {settings.ui.release}",
        tags=[settings.training.game, speedrun.game.category],
        config={
            "game": settings.training.game,
            "category": speedrun.game.category,
            "policy_emulators": speedrun.policy_emulators,
            **hyperparameters,
        },
        dir=tas_dir(settings),
        mode=ONLINE if tracked else DISABLED,
    )
