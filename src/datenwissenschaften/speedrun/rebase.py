from itertools import pairwise

from loguru import logger

from datenwissenschaften.speedrun.routes import Route, Routes, finished, still_replaying
from datenwissenschaften.speedrun.runner import SegmentRunner, Start


def rebased(
    runner: SegmentRunner, routes: Routes, name: str, current: str | None, end: bytes
) -> dict[str, Route] | None:
    names = [part.name for part in routes.parts]
    moved: dict[str, Route] = {}
    for part in routes.parts[names.index(name) + 1 :]:
        if part.name not in routes.routes:
            continue
        segments = routes.segments(part.name)
        replay = still_replaying(runner, Start(part.entry_state, end), segments)
        if part.name != current and len(replay) < len(segments):
            return None
        moved[part.name] = Route(end, replay, routes.version)
        end = replay[-1].emulator_state if replay else end
    return moved


def join(runner: SegmentRunner, routes: Routes) -> None:
    for before, part in pairwise(routes.parts):
        played = routes.segments(before.name)
        if part.name not in routes.routes or not finished(played, before.exit) or not routes.loaded(part):
            continue
        end, segments = played[-1].emulator_state, routes.segments(part.name)
        replay = still_replaying(runner, Start(part.entry_state, end), segments)
        routes.move({part.name: Route(end, replay, routes.version)})
        logger.info(
            f"{part.name} starts where {before.name} ends again; {len(replay)} of {len(segments)} pieces replay"
        )
