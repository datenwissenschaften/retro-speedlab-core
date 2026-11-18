from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datenwissenschaften.speedrun.attempt import Attempt, Lead, Origin
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import Crossing, Searched
from datenwissenschaften.speedrun.searcher import Job, Searcher
from datenwissenschaften.speedrun.workers import explore

SECONDS = 1.0


def searcher(recorded: list[Attempt]) -> Searcher:
    events = SimpleNamespace(attempt=lambda attempt, seconds: recorded.append(attempt))
    return Searcher("wrapper", Path("config.yaml"), None, None, SimpleNamespace(events=events))


def crossing(frames: int) -> Crossing:
    return Crossing(Start("Stage", bytes([frames])), np.zeros((frames, 1), dtype=np.int8), False, ("Stage", frames))


def finished_future(outcome: object) -> Future[object]:
    future: Future[object] = Future()
    if isinstance(outcome, Exception):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)
    return future


class LookAheadPool:
    def __init__(self, ahead: list[Attempt | Exception]) -> None:
        self.ahead = iter(ahead)

    def submit(self, function: object, *args: object) -> Future[object]:
        return finished_future(next(self.ahead) if function is explore else args[3])


def job(index: int, outcome: Attempt | Exception) -> Job:
    return Job(index, finished_future(outcome))


def test_a_job_that_raises_in_the_game_leaves_the_round_to_the_other_jobs():
    recorded: list[Attempt] = []
    scouted = Attempt("Stage", 7, Origin("scout", 0.5), Searched([], None))

    attempts = searcher(recorded).attempts([job(0, IndexError("room 5")), job(1, scouted)], SECONDS)

    assert attempts == [(1, scouted)]
    assert recorded == [scouted]


def test_a_round_whose_every_job_raises_stops_the_search():
    with pytest.raises(RuntimeError, match="All 2 search jobs"):
        searcher([]).attempts([job(0, IndexError("room 5")), job(1, IndexError("room 5"))], SECONDS)


def test_a_look_ahead_job_that_raises_in_the_game_counts_as_stuck_and_the_choice_goes_on():
    settings = SimpleNamespace(speedrun=SimpleNamespace(segment_seconds=SECONDS))
    after = Attempt("Stage", 1, Origin("scout", 0.5), Searched([crossing(3)], None))
    pool = LookAheadPool([IndexError("room 5"), after])
    leads = [Lead(0, crossing(2), Origin("scout", 0.5)), Lead(0, crossing(4), Origin("scout", 0.5))]

    chosen = Searcher("wrapper", Path("config.yaml"), settings, pool, None).chosen(Start("Stage", bytes([0])), leads)

    assert chosen.crossing.frames == 4
