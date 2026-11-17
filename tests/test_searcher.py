from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften.speedrun.attempt import Attempt, Origin
from datenwissenschaften.speedrun.search import Searched
from datenwissenschaften.speedrun.searcher import Job, Searcher

SECONDS = 1.0


def searcher(recorded: list[Attempt]) -> Searcher:
    events = SimpleNamespace(attempt=lambda attempt, seconds: recorded.append(attempt))
    return Searcher("wrapper", Path("config.yaml"), None, None, SimpleNamespace(events=events))


def job(index: int, outcome: Attempt | Exception) -> Job:
    future: Future[Attempt] = Future()
    if isinstance(outcome, Exception):
        future.set_exception(outcome)
    else:
        future.set_result(outcome)
    return Job(index, future)


def test_a_job_that_raises_in_the_game_leaves_the_round_to_the_other_jobs():
    recorded: list[Attempt] = []
    scouted = Attempt("Stage", 7, Origin("scout", 0.5), Searched([], None))

    attempts = searcher(recorded).attempts([job(0, IndexError("room 5")), job(1, scouted)], SECONDS)

    assert attempts == [(1, scouted)]
    assert recorded == [scouted]


def test_a_round_whose_every_job_raises_stops_the_search():
    with pytest.raises(RuntimeError, match="All 2 search jobs"):
        searcher([]).attempts([job(0, IndexError("room 5")), job(1, IndexError("room 5"))], SECONDS)
