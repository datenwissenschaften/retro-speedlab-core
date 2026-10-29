import subprocess
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

from datenwissenschaften.environment.demonstration import TAS_DIR, TAS_PROGRESS
from datenwissenschaften.environment.level_drill import DRILLED_SUFFIX
from datenwissenschaften.lab.critic import CRITIC_FILE, read_critic
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.stuck import STUCK_FILE, load_stuck
from datenwissenschaften.tracking.stale import close_stale_lab_runs
from datenwissenschaften.tracking.task_run import task_run
from datenwissenschaften.workflow.environment import speedlab

SECONDS_PER_MINUTE = 60
LAST_LAB_RUN = ".agent-last-run"
CRITIC_RUNS = 5
REPORT_STAMP = "%Y-%m-%dT%H%M"


def last_progress(config: RetroSpeedlabConfig) -> float:
    curricula = config.paths.cache_dir / "curriculum"
    wins = [*curricula.glob("*/*.successes"), *curricula.glob(f"*/*{DRILLED_SUFFIX}")]
    speedrun = config.paths.cache_dir / TAS_DIR / TAS_PROGRESS
    marks = [*wins, speedrun, config.paths.lab_run_marker.with_name(LAST_LAB_RUN)]
    return max((mark.stat().st_mtime for mark in marks if mark.is_file()), default=0.0)


@speedlab.task
def lab_run_due(config: str) -> bool:
    settings = load_config(Path(config))
    setup_logging(settings.log_level)
    if settings.paths.lab_run_marker.exists():
        logger.info("No lab run: one is running")
        return False
    close_stale_lab_runs(settings)
    stuck = load_stuck(settings.paths.cache_dir / TAS_DIR / STUCK_FILE)
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
def lab_run(config: str, project: str) -> None:
    with task_run(Path(config), "lab_run"):
        subprocess.run([Path(project) / "agent" / "daily.sh"], check=True)


def reports_since_critic(config: str, project: str) -> int:
    critic = read_critic(load_config(Path(config)).paths.lab_run_marker.with_name(CRITIC_FILE))
    stamps = [report.stem for report in (Path(project) / "agent" / "reports").glob("*.md")]
    if critic is None:
        return len(stamps)
    reviewed = datetime.fromisoformat(critic.at).strftime(REPORT_STAMP)
    return sum(stamp > reviewed for stamp in stamps)


@speedlab.task
def critic_run(project: str) -> None:
    subprocess.run([Path(project) / "agent" / "critic.sh"], check=True)


@speedlab.task
def lab(config: str, project: str) -> None:
    if lab_run_due(config):
        lab_run(config, project)
        return
    reports = reports_since_critic(config, project)
    if reports < CRITIC_RUNS:
        logger.info(f"No critic: {reports} lab reports since the last one")
        return
    logger.info(f"Critic: {reports} lab reports since the last one")
    critic_run(project)
