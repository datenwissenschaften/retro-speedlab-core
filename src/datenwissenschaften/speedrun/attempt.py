import time
from collections.abc import Callable
from dataclasses import dataclass

from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import Crossing, Searched


@dataclass(slots=True, frozen=True)
class Origin:
    role: str
    worker_seconds: float


@dataclass(slots=True, frozen=True)
class Attempt:
    state: str
    seed: int
    origin: Origin
    searched: Searched


@dataclass(slots=True, frozen=True)
class Lead:
    index: int
    crossing: Crossing
    origin: Origin


def attempted(start: Start, seed: int, role: str, search: Callable[[], Searched]) -> Attempt:
    started = time.monotonic()
    searched = search()
    return Attempt(start.state, seed, Origin(role, time.monotonic() - started), searched)
