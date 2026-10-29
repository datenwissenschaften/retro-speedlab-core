import math
import time
from dataclasses import asdict
from pathlib import Path

from loguru import logger

from datenwissenschaften.lab.activity import WORKING
from datenwissenschaften.lab.critic import CRITIC_FILE, read_critic
from datenwissenschaften.lab.marker import LabMarker, read_marker
from datenwissenschaften.lab.status import IDLE, lab_status
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset
from datenwissenschaften.ui.telemetry import publish_metadata

POLL_SECONDS = 10.0
SECTION = "lab_run"
CRITIC_SECTION = "critic"


def pausing(lab: LabMarker | None) -> bool:
    return lab is not None and lab.phase != WORKING


class LabRun:
    def __init__(self, marker: Path) -> None:
        self.marker = marker
        self.checked_at = -math.inf

    def paused(self) -> bool:
        return pausing(read_marker(self.marker))

    def wait(self) -> ModelResetRequest | None:
        if time.monotonic() - self.checked_at < POLL_SECONDS:
            return None
        self.checked_at = time.monotonic()
        lab = self.publish()
        if not pausing(lab):
            return None
        logger.info("A lab run is judging its branches; training pauses until it ends")
        while pausing(lab):
            if (request := consume_model_reset()) is not None:
                return request
            time.sleep(POLL_SECONDS)
            lab = self.publish()
        logger.info("The lab run ended; training resumes")
        return None

    def publish(self) -> LabMarker | None:
        lab = read_marker(self.marker)
        publish_metadata(SECTION, IDLE if lab is None else asdict(lab_status(lab)), replace=True)
        if (critic := read_critic(self.marker.with_name(CRITIC_FILE))) is not None:
            publish_metadata(CRITIC_SECTION, asdict(critic), replace=True)
        return lab
