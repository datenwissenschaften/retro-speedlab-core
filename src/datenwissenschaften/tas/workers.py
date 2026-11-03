import os
from functools import cache
from pathlib import Path

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.policy.prior import POLICY_FILE, PolicyPrior
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.archives import Archives
from datenwissenschaften.tas.explore import search_segment
from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.paths import tas_dir
from datenwissenschaften.tas.preferences import Preferences
from datenwissenschaften.tas.prior import PRIOR_FILE, ActionPrior
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Crossing, Searched, shave
from datenwissenschaften.tas.strategies import WALK
from datenwissenschaften.workflow.wrapper import wrapper_class


@cache
def worker_runner(wrapper: str, config: Path) -> SegmentRunner:
    return SegmentRunner(make_environment(wrapper_class(wrapper), load_config(config), os.getpid()))


@cache
def worker_preferences(config: Path) -> Preferences:
    output = tas_dir(load_config(config))
    return Preferences(ActionPrior(output / PRIOR_FILE), PolicyPrior(output / POLICY_FILE))


@cache
def worker_archives() -> Archives:
    return Archives()


def explore(
    wrapper: str, config: Path, start: Start, bound: int, seed: int, seconds: float, live: Path | None
) -> Searched:
    watched = None if live is None else WorkerLive(live, seed, WALK)
    runner, preferences = worker_runner(wrapper, config), worker_preferences(config)
    return search_segment(runner, start, bound, seed, seconds, watched, preferences, worker_archives())


def polish(wrapper: str, config: Path, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    return shave(worker_runner(wrapper, config), start, crossing, seconds)
