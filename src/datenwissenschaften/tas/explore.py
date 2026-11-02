import random
import time

from datenwissenschaften.tas.exploration import Exploration
from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.prior import ActionPrior
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Searched
from datenwissenschaften.tas.strategies import frontier, walk
from datenwissenschaften.tas.tree import tree

STRATEGIES = (walk, frontier, tree)
SHARES = (0.6, 0.2, 0.2)


def search_segment(
    runner: SegmentRunner,
    start: Start,
    bound: int,
    seed: int,
    seconds: float,
    live: WorkerLive | None,
    prior: ActionPrior,
) -> Searched:
    rng = random.Random(seed)
    exploration = Exploration(runner, start, bound, time.monotonic() + seconds, live, prior)
    spent = [0.0] * len(STRATEGIES)
    while exploration.running():
        turn = min(range(len(STRATEGIES)), key=lambda index: spent[index] / SHARES[index])
        began = time.monotonic()
        STRATEGIES[turn](exploration, rng)
        spent[turn] += time.monotonic() - began
    return exploration.searched()
