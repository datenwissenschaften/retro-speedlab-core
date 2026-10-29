import os
from functools import cache
from pathlib import Path

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.live import Watch, WorkerLive
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Crossing, search_segment, shave
from datenwissenschaften.workflow.wrapper import wrapper_class


@cache
def worker_runner(wrapper: str, config: Path) -> SegmentRunner:
    return SegmentRunner(make_environment(wrapper_class(wrapper), load_config(config), os.getpid()))


def explore(
    wrapper: str, config: Path, start: Start, bound: int, seed: int, seconds: float, watch: Watch | None
) -> list[Crossing]:
    live = None if watch is None else WorkerLive(watch, seed)
    return search_segment(worker_runner(wrapper, config), start, bound, seed, seconds, live)


def polish(wrapper: str, config: Path, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    return shave(worker_runner(wrapper, config), start, crossing, seconds)
