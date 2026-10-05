import time
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset
from datenwissenschaften.ui.telemetry import publish_metadata

POLL_SECONDS = 30.0


class LabRun:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def deadline(self) -> float | None:
        if not self.marker.is_file():
            return None
        deadline = float(self.marker.read_text(encoding="utf-8").strip())
        return deadline if deadline > time.time() else None

    def wait(self) -> ModelResetRequest | None:
        deadline = self.deadline()
        if deadline is None:
            return None
        logger.info("A lab run is working on the game; training pauses until it ends")
        while deadline is not None:
            publish_metadata("lab_run", {"active": True, "until": datetime.fromtimestamp(deadline, UTC).isoformat()})
            if (request := consume_model_reset()) is not None:
                return request
            time.sleep(POLL_SECONDS)
            deadline = self.deadline()
        publish_metadata("lab_run", {"active": False, "until": None})
        logger.info("The lab run ended; training resumes")
        return None
