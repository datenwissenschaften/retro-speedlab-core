import time
from dataclasses import dataclass, replace

from datenwissenschaften.route.route_map import left_in
from datenwissenschaften.speedrun.archives import Archive
from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.runner import CellKey, SegmentRunner, Start, route_cell, route_needs
from datenwissenschaften.speedrun.search import (
    BEAM,
    MAX_CELLS,
    Cell,
    Crossing,
    Searched,
    Trail,
    checkpoint_of,
    confirmed,
    fastest_distinct,
    leaves,
)


@dataclass(slots=True, frozen=True)
class Step:
    trail: Trail | None
    frames: int
    score: float
    ended: bool


def step_of(cell: Cell) -> Step:
    return Step(cell.trail, cell.frames, cell.score, False)


class Exploration:
    def __init__(
        self,
        runner: SegmentRunner,
        start: Start,
        bound: int,
        deadline: float,
        live: WorkerLive | None,
        preferences: Preferences,
        archive: Archive,
    ) -> None:
        self.runner = runner
        self.start = start
        self.bound = bound
        self.deadline = deadline
        self.live = live
        self.preferences = preferences
        runner.restore(start)
        self.novelty = archive.novelty
        self.archive: dict[CellKey, Cell] = archive.cells
        self.crossings: list[Crossing] = []
        self.guided = False
        self.guided_by: int | None = None
        known = [cell.remaining for cell in self.archive.values() if cell.remaining is not None]
        self.closest = min(known, default=None)

    def cells(self) -> list[Cell]:
        routes = self.runner.wrapper.routes
        route = routes.reload()
        if not self.guided or routes.loaded != self.guided_by:
            for key, cell in self.archive.items():
                cell.remaining = left_in(route, route_cell(key), route_needs(key))
            self.guided, self.guided_by = True, routes.loaded
        return list(self.archive.values())

    def running(self) -> bool:
        return time.monotonic() < self.deadline

    def limit(self) -> int:
        return self.bound if len(self.crossings) < BEAM else self.crossings[-1].frames

    def act(self, step: Step, action: int) -> Step:
        outcome = self.runner.decide(action)
        trail = Trail(step.trail, outcome.pressed)
        moved = Step(trail, step.frames + len(outcome.pressed), step.score + outcome.reward, False)
        if self.live is not None:
            best = self.crossings[0].frames if self.crossings else None
            self.live.step(self.runner, len(self.archive), best, len(outcome.pressed), self.closest)
        if moved.frames >= self.limit():
            return replace(moved, ended=True)
        if leaves(self.runner, self.start, outcome.state, outcome.won):
            if (crossing := confirmed(self.runner, trail.frames(), outcome.won)) is not None:
                self.crossings = fastest_distinct([*self.crossings, crossing])
            return replace(moved, ended=True)
        return replace(moved, ended=not outcome.alive or outcome.state != self.start.state)

    def remember(self, step: Step) -> bool:
        key = self.runner.key(step.frames)
        if key in self.archive and self.archive[key].frames <= step.frames:
            return False
        if key not in self.archive and len(self.archive) >= MAX_CELLS:
            places = [kept for kept, cell in self.archive.items() if cell.trail is not None]
            del self.archive[min(places, key=lambda kept: (self.archive[kept].score, -self.archive[kept].visits))]
        here = self.runner.here(self.start.state)
        novelty = self.novelty.see(self.runner.memory())
        remaining = self.runner.remaining()
        self.archive[key] = Cell(here, step.trail, step.frames, 0, remaining, step.score, novelty)
        if remaining is not None and (self.closest is None or remaining < self.closest):
            self.closest = remaining
        return True

    def kept(self) -> Archive:
        return Archive(self.archive, self.novelty)

    def searched(self) -> Searched:
        return Searched(self.crossings, checkpoint_of(self.archive))
