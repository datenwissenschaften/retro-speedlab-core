import random
import time

from datenwissenschaften.tas.exploration import Exploration
from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.preferences import Preferences
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
    preferences: Preferences,
) -> Searched:
    rng = random.Random(seed)
    exploration = Exploration(runner, start, bound, time.monotonic() + seconds, live, preferences)
    spent = [0.0] * len(STRATEGIES)
    while exploration.running():
        turn = min(range(len(STRATEGIES)), key=lambda index: spent[index] / SHARES[index])
        if live is not None:
            live.mode = STRATEGIES[turn].__name__
        began = time.monotonic()
        STRATEGIES[turn](exploration, rng)
        spent[turn] += time.monotonic() - began
    return exploration.searched()
