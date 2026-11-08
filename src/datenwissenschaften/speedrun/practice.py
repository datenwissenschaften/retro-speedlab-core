import json
from pathlib import Path

import numpy as np

from datenwissenschaften.route.rows import LEVEL_GAP_FRAMES, Step, route_rows
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.routes import Part, PartProgress, Route, Routes, finished
from datenwissenschaften.speedrun.runner import SegmentRunner, Start


def part_steps(runner: SegmentRunner, part: Part, route: Route, first: int) -> list[Step]:
    runner.restore(Start(part.entry_state, route.entry))
    steps: list[Step] = []
    for frame, buttons in enumerate((buttons for segment in route.segments for buttons in segment.pressed), first):
        runner.press(np.asarray(buttons, dtype=np.int8))
        if runner.wrapper.state_machine.state_name == part.exit:
            break
        position = runner.ram.position()
        if position is not None:
            steps.append((frame, position.area, position.x, position.y))
    return steps


def progress_of(runner: SegmentRunner, routes: Routes, part: Part) -> PartProgress:
    played = routes.segments(part.name)
    runner.restore(Start(played[-1].state, played[-1].emulator_state))
    return PartProgress(
        played[-1].state,
        len(played),
        sum(len(segment.pressed) for segment in played),
        finished(played, part.exit),
        routes.loaded(part),
        runner.ram.describe(),
    )


def practised(runner: SegmentRunner, routes: Routes) -> dict[str, PartProgress]:
    return {part.name: progress_of(runner, routes, part) for part in routes.parts if routes.segments(part.name)}


def write_practice_route(runner: SegmentRunner, routes: Routes, path: Path, done: dict[str, PartProgress]) -> None:
    beaten = [part for part in routes.parts if part.name in done and done[part.name].beaten]
    if not beaten:
        return
    steps: list[Step] = []
    for part in beaten:
        first = steps[-1][0] + LEVEL_GAP_FRAMES + 1 if steps else 0
        steps += part_steps(runner, part, routes.routes[part.name], first)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(route_rows(steps)).encode())
