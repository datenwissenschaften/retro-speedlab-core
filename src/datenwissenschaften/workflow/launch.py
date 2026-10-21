import os
import signal
from pathlib import Path
from types import FrameType

import flyte
from loguru import logger

from datenwissenschaften.tracking.run import kill_active_runs
from datenwissenschaften.workflow.training import training

TERMINATED = 128 + signal.SIGTERM


def terminate(signal_number: int, frame: FrameType | None) -> None:
    logger.warning("Stopped by the container; the open MLflow runs end as killed")
    kill_active_runs()
    os._exit(TERMINATED)


def run_training(wrapper: str, config: Path) -> None:
    signal.signal(signal.SIGTERM, terminate)
    flyte.init()
    flyte.run(training, wrapper=wrapper, config=str(config))
