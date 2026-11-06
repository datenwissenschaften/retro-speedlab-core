import os
from functools import cache
from pathlib import Path

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.policy.prior import POLICY_FILE, PolicyPrior
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.archives import Archives
from datenwissenschaften.speedrun.explore import search_segment
from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.paths import archives_dir, tas_dir
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.prior import PRIOR_FILE, ActionPrior
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing, Searched, shave
from datenwissenschaften.speedrun.strategies import WALK
from datenwissenschaften.workflow.wrapper import wrapper_class


@cache
def worker_runner(wrapper: str, config: Path) -> SegmentRunner:
    return SegmentRunner(make_environment(wrapper_class(wrapper), load_config(config), os.getpid()))


@cache
def worker_preferences(config: Path) -> Preferences:
    output = tas_dir(load_config(config))
    return Preferences(ActionPrior(output / PRIOR_FILE), PolicyPrior(output / POLICY_FILE))


@cache
def worker_archives(config: Path) -> Archives:
    settings = load_config(config)
    return Archives(archives_dir(settings), settings.speedrun.workers)


def explore(
    wrapper: str, config: Path, start: Start, bound: int, seed: int, seconds: float, live: Path | None
) -> Searched:
    watched = None if live is None else WorkerLive(live, seed, WALK)
    runner, preferences = worker_runner(wrapper, config), worker_preferences(config)
    return search_segment(runner, start, bound, seed, seconds, watched, preferences, worker_archives(config))


def polish(wrapper: str, config: Path, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    return shave(worker_runner(wrapper, config), start, crossing, seconds)
