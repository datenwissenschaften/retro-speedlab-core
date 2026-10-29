import random
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tas.feed import SpeedrunFeed
from datenwissenschaften.tas.live_search import search_round
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.search import UNBOUNDED, Checkpoint, Crossing, farthest, fastest_distinct
from datenwissenschaften.tas.workers import explore, polish

SEED_SPACE = 2**31


@dataclass(slots=True)
class Searcher:
    wrapper: str
    config: Path
    settings: RetroSpeedlabConfig
    pool: Executor
    feed: SpeedrunFeed

    def candidates(self, start: Start, bound: int, rounds: int) -> tuple[list[Crossing], Checkpoint | None]:
        reached = None
        for round_index in range(rounds):
            seconds = self.settings.speedrun.segment_seconds * 2**round_index
            first = random.randrange(SEED_SPACE)
            beat = None if bound == UNBOUNDED else bound
            self.feed.begin(search_round(start.state, round_index, rounds, seconds, beat))
            searches = [
                self.pool.submit(explore, *self._job(start, bound, first + index, seconds), self.feed.live)
                for index in range(self.settings.speedrun.workers)
            ]
            results = [search.result() for search in searches]
            found = fastest_distinct([crossing for result in results for crossing in result.crossings])
            reached = farthest([reached, *(result.checkpoint for result in results)])
            self.feed.end()
            if found:
                return found, None
        return [], reached

    def best(self, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        found, reached = self.candidates(start, bound, rounds)
        if not found:
            return reached
        seconds = self.settings.speedrun.segment_seconds
        ahead = [
            self.pool.submit(explore, *self._job(crossing.end, UNBOUNDED, seed, seconds), None)
            for seed, crossing in enumerate(found)
        ]
        following = [min((after.frames for after in search.result().crossings), default=UNBOUNDED) for search in ahead]
        totals = [
            crossing.frames + (0 if crossing.won else after) for crossing, after in zip(found, following, strict=True)
        ]
        chosen = found[totals.index(min(totals))]
        return self.pool.submit(polish, self.wrapper, self.config, start, chosen, seconds).result()

    def _job(self, start: Start, bound: int, seed: int, seconds: float) -> tuple[str, Path, Start, int, int, float]:
        return self.wrapper, self.config, start, bound, seed, seconds
