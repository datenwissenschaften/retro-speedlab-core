import json
from pathlib import Path

import msgspec
import numpy as np
from loguru import logger

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.reference.trace import LEVEL_GAP_FRAMES, Step, route_rows
from datenwissenschaften.route.position import Position
from datenwissenschaften.tas.paths import atomic_write
from datenwissenschaften.tas.routes import Route, Routes, finished
from datenwissenschaften.tas.runner import SegmentRunner, Start

LOADER_FILE = "loader.msgpack"
ARRIVAL_FRAMES = 600


class LevelLoader(msgspec.Struct, frozen=True):
    state: bytes
    delta: int


def changed_level(before: RamInfo, after: RamInfo) -> bool:
    levels = (before.level(), after.level())
    return before.position() is not None and None not in levels and levels[0] != levels[1]


def loader_in(runner: SegmentRunner, route: Route) -> LevelLoader | None:
    runner.restore(Start(runner.order[0], route.entry))
    for segment in route.segments:
        for buttons in segment.pressed:
            state, before = runner.snapshot(), runner.ram
            runner.press(np.asarray(buttons, dtype=np.int8))
            if changed_level(before, runner.ram):
                return LevelLoader(state, int(runner.ram.level()) - int(before.level()))
    return None


def ensure_loader(path: Path, runner: SegmentRunner, routes: Routes) -> LevelLoader | None:
    if path.is_file():
        return msgspec.msgpack.decode(path.read_bytes(), type=LevelLoader)
    if runner.wrapper.level_address is None:
        return None
    for name, route in routes.routes.items():
        loader = loader_in(runner, route)
        if loader is not None:
            atomic_write(path, msgspec.msgpack.encode(loader))
            logger.info(f"The route {name} changes the level by {loader.delta}: every level can be practised now")
            return loader
    logger.info("No stored route changes the level yet, so the levels are practised in order")
    return None


def moved(departed: Position | None, arrived: Position | None) -> bool:
    return arrived is not None and (departed is None or (arrived.x, arrived.y) != (departed.x, departed.y))


def level_start(runner: SegmentRunner, loader: LevelLoader, level: int, state: str) -> Start:
    runner.restore(Start(runner.order[0], loader.state))
    departed = runner.ram.position()
    runner.set_level(level - loader.delta)
    idle = np.zeros(len(runner.emulator.buttons), dtype=np.int8)
    for _ in range(ARRIVAL_FRAMES):
        runner.press(idle)
        arrived = runner.ram.position()
        if runner.ram.level() == level and moved(departed, arrived):
            return Start(state, runner.snapshot())
    raise RuntimeError(f"The level loader brought level {level} into no play within {ARRIVAL_FRAMES} frames")


def part_steps(runner: SegmentRunner, entry: Start, route: Route, first: int) -> list[Step]:
    runner.restore(entry)
    steps: list[Step] = []
    for frame, buttons in enumerate((buttons for segment in route.segments for buttons in segment.pressed), first):
        runner.press(np.asarray(buttons, dtype=np.int8))
        position = runner.ram.position()
        if position is not None:
            steps.append((frame, position.area, position.x, position.y))
    return steps


def practised(routes: Routes) -> dict[str, int]:
    done = {part.name: routes.segments(part.name) for part in routes.parts}
    return {
        part.name: sum(len(segment.pressed) for segment in done[part.name])
        for part in routes.parts
        if finished(done[part.name], part.exit)
    }


def write_practice_route(runner: SegmentRunner, routes: Routes, path: Path) -> None:
    steps: list[Step] = []
    done = practised(routes)
    for part in routes.parts:
        if part.name in done:
            route = routes.routes[part.name]
            first = steps[-1][0] + LEVEL_GAP_FRAMES + 1 if steps else 0
            steps += part_steps(runner, Start(part.entry_state, route.entry), route, first)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(route_rows(steps)).encode())
