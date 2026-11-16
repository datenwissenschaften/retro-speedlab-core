import random
from dataclasses import dataclass

import numpy as np
from loguru import logger

from datenwissenschaften.speedrun.attempt import Lead, Origin
from datenwissenschaften.speedrun.movie import Segment, frames
from datenwissenschaften.speedrun.rebase import rebased
from datenwissenschaften.speedrun.routes import Leg, Part, finished, still_replaying
from datenwissenschaften.speedrun.run import Tas, segment_of
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

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


@dataclass(slots=True)
class Polisher:
    tas: Tas
    searches: int

    def rest(self, current: Part) -> bool:
        self.searches += 1
        routes = self.tas.routes
        beaten = [part for part in routes.parts if part != current and finished(routes.segments(part.name), part.exit)]
        if self.searches % POLISH_EVERY or not beaten:
            return False
        part = max(beaten, key=lambda each: frames(routes.segments(each.name)))
        segments = routes.segments(part.name)
        cut = random.choice(cut_points(frames(segments)))
        leg = Leg(part.name, Start(part.entry_state, routes.routes[part.name].entry), part.exit)
        faster = self.tail(leg, segments, cut)
        return faster is not None and self.commit(part, current, leg, segments, *faster)

    def tail(self, leg: Leg, segments: list[Segment], cut: int) -> tuple[list[Segment], Origin] | None:
        head, start = played_until(self.tas.runner, leg.entry, segments, cut)
        bound = frames(segments) - cut - SMALLEST_GAIN_FRAMES
        found = self.tas.searcher.best(start, bound, TAIL_ROUNDS)
        if not isinstance(found, Lead) or found.crossing.frames > bound or found.crossing.end.state != leg.exit:
            logger.info(f"{leg.name} from frame {cut}: no way to its exit in {bound} frames or fewer")
            return None
        return [*head, segment_of(found.crossing)], found.origin

    def commit(
        self, part: Part, current: Part, leg: Leg, segments: list[Segment], faster: list[Segment], origin: Origin
    ) -> bool:
        routes, runner = self.tas.routes, self.tas.runner
        replay = still_replaying(runner, leg.entry, faster)
        if len(replay) < len(faster):
            return False
        saved = frames(segments) - frames(replay)
        moved = rebased(runner, routes, part.name, current.name, replay[-1].emulator_state)
        if moved is None:
            logger.info(f"{part.name} is {saved} frames faster but a beaten level after it no longer replays; it stays")
            return False
        routes.save(leg, replay)
        routes.move(moved)
        kept = len(moved[current.name].segments) if current.name in moved else 0
        logger.info(f"{part.name} is {saved} frames faster; {current.name} keeps {kept} pieces from its new start")
        self.tas.feed.event("improved", replay[-1].state, frames(replay), saved, origin)
        return True
