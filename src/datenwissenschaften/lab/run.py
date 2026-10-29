from dataclasses import asdict
from pathlib import Path

from datenwissenschaften.lab.marker import read_marker
from datenwissenschaften.lab.status import IDLE, lab_status
from datenwissenschaften.ui.telemetry import publish_metadata

SECTION = "lab_run"


def publish_lab_run(marker: Path) -> None:
    lab = read_marker(marker)
    publish_metadata(SECTION, IDLE if lab is None else asdict(lab_status(lab)))
