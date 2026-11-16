import random
from concurrent.futures import Executor, Future
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.attempt import Attempt, Lead
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.live import BACKTRACKER, SCOUT
from datenwissenschaften.speedrun.live_search import search_round
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Checkpoint, farthest, fastest_distinct
from datenwissenschaften.speedrun.workers import beam, explore, polish, vision

SEED_SPACE = 2**31


@dataclass(slots=True, frozen=True)
class Job:
    index: int
    future: Future[Attempt]


@dataclass(slots=True)
class Searcher:
    wrapper: str
    config: Path
    settings: RetroSpeedlabConfig
    pool: Executor
    feed: SpeedrunFeed

    def round_leads(self, starts: list[Start], bound: int, rounds: int) -> tuple[list[Lead], Checkpoint | None]:
        reached = None
        for round_index in range(rounds):
            seconds = self.settings.speedrun.segment_seconds * 2**round_index
            beat = None if bound == UNBOUNDED else bound
            self.feed.begin(search_round(starts[-1].state, round_index, rounds, seconds, beat))
            attempts = [(job.index, self.finished(job, seconds)) for job in self.jobs(starts, bound, seconds)]
            self.feed.end()
            leads = [
                Lead(index, crossing, attempt.origin)
                for index, attempt in attempts
                for crossing in attempt.searched.crossings
            ]
            found = fastest_distinct([lead.crossing for lead in leads])
            frontier = [attempt.searched.checkpoint for index, attempt in attempts if index == len(starts) - 1]
            reached = farthest([reached, *frontier])
            if found:
                return [lead for lead in leads if any(lead.crossing is fast for fast in found)], None
        return [], reached

    def finished(self, job: Job, seconds: float) -> Attempt:
        attempt = job.future.result()
        self.feed.events.attempt(attempt, seconds)
        return attempt

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
            jobs.append(Job(last, self.pool.submit(beam, self.wrapper, self.config, starts[last], seed, seconds, live)))
        for seed in islice(seeds, speedrun.vision_scouts):
            jobs.append(Job(last, self.pool.submit(vision, self.wrapper, self.config, starts[last], seed, live)))
        return jobs

    def scout(self, starts: list[Start], index: int, bound: int, seed: int, seconds: float, role: str) -> Job:
        job = self._job(starts[index], bound, seed, seconds)
        return Job(index, self.pool.submit(explore, *job, self.feed.live, role))

    def best(self, start: Start, bound: int, rounds: int) -> Lead | Checkpoint | None:
        found, reached = self.round_leads([start], bound, rounds)
        return reached if not found else self.chosen(start, found)

    def lead(self, starts: list[Start], rounds: int) -> Lead | Checkpoint | None:
        if len(starts) == 1 or not self.settings.speedrun.backtrackers:
            found = self.best(starts[-1], UNBOUNDED, rounds)
            return replace(found, index=len(starts) - 1) if isinstance(found, Lead) else found
        leads, reached = self.round_leads(starts, UNBOUNDED, rounds)
        if not leads:
            return reached
        index = max(lead.index for lead in leads)
        return self.chosen(starts[index], [lead for lead in leads if lead.index == index])

    def chosen(self, start: Start, found: list[Lead]) -> Lead:
        seconds = self.settings.speedrun.segment_seconds
        ahead = [
            self.pool.submit(explore, *self._job(lead.crossing.end, UNBOUNDED, seed, seconds), None, SCOUT)
            for seed, lead in enumerate(found)
        ]
        following = [
            min((after.frames for after in search.result().searched.crossings), default=UNBOUNDED) for search in ahead
        ]
        totals = [
            lead.crossing.frames + (0 if lead.crossing.won else after)
            for lead, after in zip(found, following, strict=True)
        ]
        chosen = found[totals.index(min(totals))]
        polished = self.pool.submit(polish, self.wrapper, self.config, start, chosen.crossing, seconds).result()
        return replace(chosen, crossing=polished)

    def _job(self, start: Start, bound: int, seed: int, seconds: float) -> tuple[str, Path, Start, int, int, float]:
        return self.wrapper, self.config, start, bound, seed, seconds
