import subprocess
import time
from pathlib import Path

from loguru import logger

from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.paths import PROGRESS_FILE, tas_dir
from datenwissenschaften.tas.stuck import STUCK_FILE, load_stuck
from datenwissenschaften.workflow.environment import speedlab

SECONDS_PER_MINUTE = 60
LAST_LAB_RUN = ".agent-last-run"


def last_progress(config: RetroSpeedlabConfig) -> float:
    marks = [tas_dir(config) / PROGRESS_FILE, config.paths.lab_run_marker.with_name(LAST_LAB_RUN)]
    return max((mark.stat().st_mtime for mark in marks if mark.is_file()), default=0.0)


@speedlab.task
def lab_run_due(config: str) -> bool:
    settings = load_config(Path(config))
    setup_logging(settings.log_level)
    if settings.paths.lab_run_marker.exists():
        logger.info("No lab run: one is running")
        return False
    stuck = load_stuck(tas_dir(settings) / STUCK_FILE)
    if stuck is not None:
        logger.info(f"Lab run: the search is stuck in {stuck.state} after {stuck.searches} searches")
        return True
    stalled_minutes = (time.time() - last_progress(settings)) / SECONDS_PER_MINUTE
    if stalled_minutes < settings.lab.stall_minutes:
        logger.info(f"No lab run: the last progress was {stalled_minutes:.0f} minutes ago")
        return False
    logger.info(f"Lab run: no progress for {stalled_minutes:.0f} minutes")
    return True


@speedlab.task
def lab_run(project: str) -> None:
    subprocess.run([Path(project) / "agent" / "daily.sh"], check=True)


@speedlab.task
def lab(config: str, project: str) -> None:
    if lab_run_due(config):
        lab_run(project)
