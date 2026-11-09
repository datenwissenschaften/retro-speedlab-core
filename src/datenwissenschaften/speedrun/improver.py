import random
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field
from pathlib import Path

from datenwissenschaften.speedrun.live import POLISHER
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import Leg
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import Crossing, Searched
from datenwissenschaften.speedrun.workers import explore

SEED_SPACE = 2**31
SMALLEST_GAIN_FRAMES = 30


@dataclass(slots=True, frozen=True)
class Target:
    leg: str
    index: int
    start: Start
    piece: Segment


def piece_start(leg: Leg, segments: list[Segment], index: int) -> Start:
    if index == 0:
        return leg.entry
    return Start(segments[index - 1].state, segments[index - 1].emulator_state)


def shortenable(leg: Leg, segments: list[Segment]) -> list[int]:
    origins = [leg.entry.state, *(segment.state for segment in segments)]
    return [
        index
        for index, segment in enumerate(segments)
        if segment.state != origins[index] and len(segment.pressed) > SMALLEST_GAIN_FRAMES
    ]


def still_there(target: Target, leg: Leg, segments: list[Segment]) -> bool:
    if target.leg != leg.name or target.index >= len(segments):
        return False
    piece = segments[target.index]
    same_start = piece_start(leg, segments, target.index) == target.start
    return same_start and piece.state == target.piece.state and piece.pressed == target.piece.pressed


def faster_crossing(target: Target, searched: Searched) -> Crossing | None:
    limit = len(target.piece.pressed) - SMALLEST_GAIN_FRAMES
    fitting = [c for c in searched.crossings if c.end.state == target.piece.state and c.frames <= limit]
    return min(fitting, key=lambda crossing: crossing.frames, default=None)


@dataclass(slots=True)
class Improver:
    wrapper: str
    config: Path
    pool: Executor
    polishers: int
    seconds: float
    running: dict[Future[Searched], Target] = field(default_factory=dict[Future[Searched], Target])

    def keep_busy(self, leg: Leg, segments: list[Segment]) -> None:
        candidates = shortenable(leg, segments)
        while candidates and len(self.running) < self.polishers:
            index = random.choice(candidates)
            target = Target(leg.name, index, piece_start(leg, segments, index), segments[index])
            bound = len(target.piece.pressed) - SMALLEST_GAIN_FRAMES
            job = (
                self.wrapper,
                self.config,
                target.start,
                bound,
                random.randrange(SEED_SPACE),
                self.seconds,
                None,
                POLISHER,
            )
            self.running[self.pool.submit(explore, *job)] = target

    def found(self, leg: Leg, segments: list[Segment]) -> list[tuple[int, Crossing]]:
        faster: list[tuple[int, Crossing]] = []
        for future in [future for future in self.running if future.done()]:
            target = self.running.pop(future)
            if not still_there(target, leg, segments):
                continue
            crossing = faster_crossing(target, future.result())
            if crossing is not None:
                faster.append((target.index, crossing))
        return faster
