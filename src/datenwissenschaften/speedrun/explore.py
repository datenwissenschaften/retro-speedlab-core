import random
import time

from datenwissenschaften.speedrun.archives import Archives
from datenwissenschaften.speedrun.exploration import Exploration
from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Searched
from datenwissenschaften.speedrun.strategies import frontier, walk
from datenwissenschaften.speedrun.tree import tree

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
    archives: Archives,
) -> Searched:
    rng = random.Random(seed)
    archive = archives.resumed(runner, start, bound)
    exploration = Exploration(runner, start, bound, time.monotonic() + seconds, live, preferences, archive)
    spent = [0.0] * len(STRATEGIES)
    while exploration.running():
        turn = min(range(len(STRATEGIES)), key=lambda index: spent[index] / SHARES[index])
        if live is not None:
            live.mode = STRATEGIES[turn].__name__
        began = time.monotonic()
        STRATEGIES[turn](exploration, rng)
        spent[turn] += time.monotonic() - began
    archives.keep(start, bound, exploration.kept())
    return exploration.searched()
