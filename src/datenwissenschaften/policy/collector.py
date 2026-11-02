from functools import lru_cache
from pathlib import Path

import numpy as np

from datenwissenschaften.policy.prior import Weights
from datenwissenschaften.policy.rollout import Rollout, collect
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.workers import worker_runner

LEVELS_REMEMBERED = 2


@lru_cache(maxsize=LEVELS_REMEMBERED)
def level_novelty(wrapper: str, config: Path, start: Start) -> RamNovelty:
    runner = worker_runner(wrapper, config)
    runner.restore(start)
    return RamNovelty(runner.memory())


def collect_rollout(wrapper: str, config: Path, start: Start, weights: Weights, decisions: int, seed: int) -> Rollout:
    novelty = level_novelty(wrapper, config, start)
    return collect(worker_runner(wrapper, config), start, weights, novelty, decisions, np.random.default_rng(seed))
