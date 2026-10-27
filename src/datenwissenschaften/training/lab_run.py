import time
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from datenwissenschaften.lab.marker import read_marker
from datenwissenschaften.lab.status import IDLE, lab_status
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset
from datenwissenschaften.ui.telemetry import publish_metadata

POLL_SECONDS = 10.0
SECTION = "lab_run"


class LabRun:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def deadline(self) -> float | None:
        lab = read_marker(self.marker)
        return None if lab is None else lab.until

    def wait(self) -> ModelResetRequest | None:
        lab = read_marker(self.marker)
        if lab is None:
            publish_metadata(SECTION, IDLE, replace=True)
            return None
        logger.info("A lab run is working on the game; training pauses until it ends")
        while lab is not None:
            publish_metadata(SECTION, asdict(lab_status(lab)), replace=True)
            if (request := consume_model_reset()) is not None:
                return request
            time.sleep(POLL_SECONDS)
            lab = read_marker(self.marker)
        publish_metadata(SECTION, IDLE, replace=True)
        logger.info("The lab run ended; training resumes")
        return None
