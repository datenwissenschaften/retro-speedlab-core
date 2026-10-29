import os
import signal
import threading
from pathlib import Path
from types import FrameType

import flyte
from flyte import TaskTemplate
from loguru import logger

from datenwissenschaften.tracking.run import kill_active_runs

TERMINATED = 128 + signal.SIGTERM
CLOSE_SECONDS = 20.0


def close_runs() -> None:
    logger.warning("Stopped by the container; the open MLflow runs end as killed")
    kill_active_runs()


def terminate(signal_number: int, frame: FrameType | None) -> None:
    closer = threading.Thread(target=close_runs, name="close-runs", daemon=True)
    closer.start()
    closer.join(CLOSE_SECONDS)
    os._exit(TERMINATED)


def launch(task: TaskTemplate, wrapper: str, config: Path) -> None:
    signal.signal(signal.SIGTERM, terminate)
    flyte.init()
    flyte.run(task, wrapper=wrapper, config=str(config))
