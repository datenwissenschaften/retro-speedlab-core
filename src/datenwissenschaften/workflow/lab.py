import subprocess
import time
from datetime import datetime
from pathlib import Path

import msgspec
from loguru import logger

from datenwissenschaften.lab.critic import CRITIC_FILE, read_critic
from datenwissenschaften.lab.reports import publish_reports
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.reference.fetch import ensure_reference
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.live import LIVE_DIR
from datenwissenschaften.tas.paths import PROGRESS_FILE, atomic_write, tas_dir
from datenwissenschaften.tas.run import BACKTRACK_SEARCHES
from datenwissenschaften.tas.stuck import STUCK_FILE, Stuck, load_stuck
from datenwissenschaften.workflow.environment import speedlab

SECONDS_PER_MINUTE = 60
LAST_LAB_RUN = ".agent-last-run"
LAB_BASELINE = ".lab-baseline.json"
CRITIC_RUNS = 5
REPORT_STAMP = "%Y-%m-%dT%H%M"


def search_running(config: RetroSpeedlabConfig) -> bool:
    live = tas_dir(config) / LIVE_DIR
    if not live.is_dir():
        return False
    return (time.time() - live.stat().st_mtime) / SECONDS_PER_MINUTE < config.lab.stall_minutes


def baseline_path(config: RetroSpeedlabConfig) -> Path:
    return config.paths.lab_run_marker.with_name(LAB_BASELINE)


def searches_since_lab(stuck: Stuck, baseline: Path) -> int:
    before = load_stuck(baseline)
    if before is None or before.state != stuck.state or before.searches > stuck.searches:
        return stuck.searches
    return stuck.searches - before.searches


def remember_baseline(config: RetroSpeedlabConfig) -> None:
    stuck = load_stuck(tas_dir(config) / STUCK_FILE)
    if stuck is None:
        baseline_path(config).unlink(missing_ok=True)
        return
    atomic_write(baseline_path(config), msgspec.json.encode(stuck))


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
    if not search_running(settings):
        logger.info("No lab run: the search is not running")
        return False
    stuck = load_stuck(tas_dir(settings) / STUCK_FILE)
    if stuck is not None:
        searches = searches_since_lab(stuck, baseline_path(settings))
        if searches < BACKTRACK_SEARCHES:
            logger.info(f"No lab run: the search tried {stuck.state} {searches} times since the last lab run")
            return False
        logger.info(f"Lab run: the search is stuck in {stuck.state} after {searches} searches since the last lab run")
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
    publish_reports(load_config(Path(config)), Path(project))
    reports = reports_since_critic(config, project)
    if reports >= CRITIC_RUNS:
        logger.info(f"Critic: {reports} lab reports since the last one")
        critic_run(project)
    if lab_run_due(config):
        settings = load_config(Path(config))
        remember_baseline(settings)
        ensure_reference(settings)
        lab_run(project)
