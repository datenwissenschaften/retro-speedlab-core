from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from loguru import logger

from datenwissenschaften.speedrun.movie import Segment, frames, verified_segments
from datenwissenschaften.speedrun.routes import Leg, Part, Route, Routes, finished
from datenwissenschaften.speedrun.run import TRIAL_MOVIE, Tas, segment_of
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing

POLISH_EVERY = 3
TAIL_ROUNDS = 1
SMALLEST_GAIN_FRAMES = 60
CUT_FRAMES = 1200


def cut_points(total: int) -> list[int]:
    return list(range(total - CUT_FRAMES, -1, -CUT_FRAMES)) or [0]


def played_until(runner: SegmentRunner, entry: Start, segments: list[Segment], cut: int) -> tuple[list[Segment], Start]:
    runner.restore(entry)
    head: list[Segment] = []
    played = 0
    for segment in segments:
        pressed = segment.pressed[: cut - played]
        for buttons in pressed:
            runner.press(np.asarray(buttons, dtype=np.int8))
        played += len(pressed)
        if not pressed:
            break
        whole = len(pressed) == len(segment.pressed)
        state = runner.wrapper.state_machine.state_name
        head.append(segment if whole else Segment(state, pressed, runner.snapshot(), False))
    return head, Start(runner.wrapper.state_machine.state_name, runner.snapshot())


def replayed(runner: SegmentRunner, entry: Start, segments: list[Segment], movie: Path) -> list[Segment]:
    replay = verified_segments(runner, entry, segments, movie, None)
    return [
        replace(segment, emulator_state=end)
        for segment, end in zip(segments[: replay.verified], replay.ends, strict=True)
    ]


def rebased(
    runner: SegmentRunner, routes: Routes, later: list[Part], current: Part, end: bytes, movie: Path
) -> dict[str, Route] | None:
    moved: dict[str, Route] = {}
    for part in later:
        segments = routes.segments(part.name)
        replay = replayed(runner, Start(part.entry_state, end), segments, movie)
        if part != current and len(replay) < len(segments):
            return None
        moved[part.name] = Route(end, replay)
        end = replay[-1].emulator_state if replay else end
    return moved


@dataclass(slots=True)
class Polisher:
    tas: Tas
    searches: int
    turns: int

    @property
    def movie(self) -> Path:
        return self.tas.output / TRIAL_MOVIE

    def rest(self, current: Part) -> bool:
        self.searches += 1
        routes = self.tas.routes
        beaten = [part for part in routes.parts if part != current and finished(routes.segments(part.name), part.exit)]
        if self.searches % POLISH_EVERY or not beaten:
            return False
        part = beaten[self.turns % len(beaten)]
        segments = routes.segments(part.name)
        cuts = cut_points(frames(segments))
        cut = cuts[self.turns // len(beaten) % len(cuts)]
        self.turns += 1
        leg = Leg(part.name, Start(part.entry_state, routes.routes[part.name].entry), part.exit)
        faster = self.tail(leg, segments, cut)
        return faster is not None and self.commit(part, current, leg, segments, faster)

    def tail(self, leg: Leg, segments: list[Segment], cut: int) -> list[Segment] | None:
        head, start = played_until(self.tas.runner, leg.entry, segments, cut)
        bound = frames(segments) - cut - SMALLEST_GAIN_FRAMES
        crossing = self.tas.searcher.best(start, bound, TAIL_ROUNDS)
        if not isinstance(crossing, Crossing) or crossing.frames > bound or crossing.end.state != leg.exit:
            logger.info(f"{leg.name} from frame {cut}: no way to its exit in {bound} frames or fewer")
            return None
        return [*head, segment_of(crossing)]

    def commit(self, part: Part, current: Part, leg: Leg, segments: list[Segment], faster: list[Segment]) -> bool:
        routes, runner = self.tas.routes, self.tas.runner
        replay = replayed(runner, leg.entry, faster, self.movie)
        if len(replay) < len(faster):
            return False
        saved = frames(segments) - frames(replay)
        later = [each for each in routes.parts[routes.parts.index(part) + 1 :] if each.name in routes.routes]
        moved = rebased(runner, routes, later, current, replay[-1].emulator_state, self.movie)
        if moved is None:
            logger.info(f"{part.name} is {saved} frames faster but a beaten level after it no longer replays; it stays")
            return False
        routes.save(leg, replay)
        for each in later:
            route = moved[each.name]
            routes.save(Leg(each.name, Start(each.entry_state, route.entry), each.exit), list(route.segments))
        kept = len(moved[current.name].segments) if current.name in moved else 0
        logger.info(f"{part.name} is {saved} frames faster; {current.name} keeps {kept} pieces from its new start")
        self.tas.feed.event("improved", replay[-1].state, frames(replay))
        self.tas.feed.stats.trimmed(saved)
        return True
