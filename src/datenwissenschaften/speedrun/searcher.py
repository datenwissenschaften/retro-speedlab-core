import random
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.live import BACKTRACKER, BEAM_SCOUT, SCOUT, VISION_SCOUT
from datenwissenschaften.speedrun.live_search import search_round
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Checkpoint, Crossing, Searched, farthest, fastest_distinct
from datenwissenschaften.speedrun.workers import beam, explore, polish, vision

SEED_SPACE = 2**31


@dataclass(slots=True, frozen=True)
class Job:
    index: int
    seed: int
    role: str
    future: Future[Searched]


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
            results = [(job.index, self.finished(starts, job, seconds)) for job in jobs]
            self.feed.end()
            found = fastest_distinct([crossing for _, result in results for crossing in result.crossings])
            leads = [Lead(index, crossing) for index, result in results for crossing in result.crossings]
            frontier = [result.checkpoint for index, result in results if index == len(starts) - 1]
            reached = farthest([reached, *frontier])
            if found:
                return [lead for lead in leads if any(lead.crossing is fast for fast in found)], None
        return [], reached

    def finished(self, starts: list[Start], job: Job, seconds: float) -> Searched:
        searched = job.future.result()
        self.feed.events.attempt(starts[job.index].state, job.seed, job.role, seconds, searched)
        return searched

    def jobs(self, starts: list[Start], bound: int, seconds: float) -> list[Job]:
        speedrun, last = self.settings.speedrun, len(starts) - 1
        seeds = iter(range(random.randrange(SEED_SPACE), SEED_SPACE * 2))
        jobs = [self.scout(starts, last, bound, next(seeds), seconds, SCOUT) for _ in range(speedrun.scouts)]
        if bound != UNBOUNDED:
            return jobs
        origins = [random.randrange(last) if last else last for _ in range(speedrun.backtrackers)]
        jobs += [self.scout(starts, origin, bound, next(seeds), seconds, BACKTRACKER) for origin in origins]
        live = self.feed.live
        for seed in islice(seeds, speedrun.beam_scouts):
            searched = self.pool.submit(beam, self.wrapper, self.config, starts[last], seed, seconds, live)
            jobs.append(Job(last, seed, BEAM_SCOUT, searched))
        for seed in islice(seeds, speedrun.vision_scouts):
            searched = self.pool.submit(vision, self.wrapper, self.config, starts[last], seed, live)
            jobs.append(Job(last, seed, VISION_SCOUT, searched))
        return jobs

    def scout(self, starts: list[Start], index: int, bound: int, seed: int, seconds: float, role: str) -> Job:
        searched = self.pool.submit(explore, *self._job(starts[index], bound, seed, seconds), self.feed.live, role)
        return Job(index, seed, role, searched)

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
            self.pool.submit(explore, *self._job(crossing.end, UNBOUNDED, seed, seconds), None, SCOUT)
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
