import random
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.live_search import search_round
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Checkpoint, Crossing, Searched, farthest, fastest_distinct
from datenwissenschaften.speedrun.workers import beam, explore, polish, vision

SEED_SPACE = 2**31


@dataclass(slots=True, frozen=True)
class Lead:
    index: int
    crossing: Crossing


@dataclass(slots=True)
class Searcher:
    wrapper: str
    config: Path
    settings: RetroSpeedlabConfig
    pool: Executor
    feed: SpeedrunFeed

    def candidates(self, start: Start, bound: int, rounds: int) -> tuple[list[Crossing], Checkpoint | None]:
        leads, reached = self.round_leads([start], bound, rounds)
        return [lead.crossing for lead in leads], reached

    def round_leads(self, starts: list[Start], bound: int, rounds: int) -> tuple[list[Lead], Checkpoint | None]:
        reached = None
        for round_index in range(rounds):
            seconds = self.settings.speedrun.segment_seconds * 2**round_index
            beat = None if bound == UNBOUNDED else bound
            self.feed.begin(search_round(starts[-1].state, round_index, rounds, seconds, beat))
            jobs = self.jobs(starts, bound, seconds)
            results = [(index, job.result()) for index, job in jobs]
            self.feed.end()
            found = fastest_distinct([crossing for _, result in results for crossing in result.crossings])
            leads = [Lead(index, crossing) for index, result in results for crossing in result.crossings]
            frontier = [result.checkpoint for index, result in results if index == len(starts) - 1]
            reached = farthest([reached, *frontier])
            if found:
                return [lead for lead in leads if any(lead.crossing is fast for fast in found)], None
        return [], reached

    def jobs(self, starts: list[Start], bound: int, seconds: float) -> list[tuple[int, Future[Searched]]]:
        speedrun, last = self.settings.speedrun, len(starts) - 1
        seeds = iter(range(random.randrange(SEED_SPACE), SEED_SPACE * 2))
        jobs = [(last, self.scout(starts[last], bound, next(seeds), seconds)) for _ in range(speedrun.scouts)]
        if bound != UNBOUNDED:
            return jobs
        origins = [random.randrange(last) if last else last for _ in range(speedrun.backtrackers)]
        jobs += [(origin, self.scout(starts[origin], bound, next(seeds), seconds)) for origin in origins]
        live = self.feed.live
        jobs += [
            (last, self.pool.submit(beam, self.wrapper, self.config, starts[last], next(seeds), seconds, live))
            for _ in range(speedrun.beam_scouts)
        ]
        jobs += [
            (last, self.pool.submit(vision, self.wrapper, self.config, starts[last]))
            for _ in range(speedrun.vision_scouts)
        ]
        return jobs

    def scout(self, start: Start, bound: int, seed: int, seconds: float) -> Future[Searched]:
        return self.pool.submit(explore, *self._job(start, bound, seed, seconds), self.feed.live)

    def best(self, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        found, reached = self.candidates(start, bound, rounds)
        return reached if not found else self.chosen(start, found)

    def lead(self, starts: list[Start], rounds: int) -> Lead | Checkpoint | None:
        if len(starts) == 1 or not self.settings.speedrun.backtrackers:
            found = self.best(starts[-1], UNBOUNDED, rounds)
            return Lead(len(starts) - 1, found) if isinstance(found, Crossing) else found
        leads, reached = self.round_leads(starts, UNBOUNDED, rounds)
        if not leads:
            return reached
        index = max(lead.index for lead in leads)
        return Lead(index, self.chosen(starts[index], [lead.crossing for lead in leads if lead.index == index]))

    def chosen(self, start: Start, found: list[Crossing]) -> Crossing:
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
