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


@dataclass(slots=True, frozen=True)
class Repair:
    leg: str
    original: list[Segment]
    chain: list[Segment]


def same_pieces(kept: list[Segment], segments: list[Segment]) -> bool:
    current = segments[: len(kept)]
    return len(current) == len(kept) and all(
        piece.state == now.state and piece.pressed == now.pressed for piece, now in zip(kept, current, strict=True)
    )


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
    live: Path | None
    running: dict[Future[Searched], Target] = field(default_factory=dict[Future[Searched], Target])
    repairs: dict[Future[Searched], tuple[Repair, int]] = field(
        default_factory=dict[Future[Searched], tuple[Repair, int]]
    )

    def keep_busy(self, leg: Leg, segments: list[Segment]) -> None:
        candidates = shortenable(leg, segments)
        while candidates and len(self.running) + len(self.repairs) < self.polishers:
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
                self.live,
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

    def repair(self, repair: Repair, bound: int) -> None:
        start = Start(repair.chain[-1].state, repair.chain[-1].emulator_state)
        job = (self.wrapper, self.config, start, bound, random.randrange(SEED_SPACE), self.seconds, self.live, POLISHER)
        self.repairs[self.pool.submit(explore, *job)] = (repair, bound)

    def repaired(self, leg: Leg, segments: list[Segment]) -> list[tuple[Repair, Crossing]]:
        rebuilt: list[tuple[Repair, Crossing]] = []
        for future in [future for future in self.repairs if future.done()]:
            repair, bound = self.repairs.pop(future)
            if repair.leg != leg.name or not same_pieces(repair.original, segments):
                continue
            target = segments[len(repair.chain)].state
            fitting = [c for c in future.result().crossings if c.end.state == target and c.frames <= bound]
            if fitting:
                rebuilt.append((repair, min(fitting, key=lambda crossing: crossing.frames)))
        return rebuilt
