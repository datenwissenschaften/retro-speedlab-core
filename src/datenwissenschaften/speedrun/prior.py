import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

import msgspec
import numpy as np

from datenwissenschaften.route.route_map import changed_at
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.routes import Routes
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

PRIOR_FILE = "prior.msgpack"

Table = dict[str, dict[int, int]]


def matching_action(table: np.ndarray, chunk: np.ndarray) -> int | None:
    if len(chunk) != table.shape[1]:
        return None
    return next((action for action in range(len(table)) if np.array_equal(table[action], chunk)), None)


def decisions(runner: SegmentRunner, start: Start, pressed: np.ndarray) -> Iterator[tuple[tuple[str, str], int]]:
    runner.restore(start)
    index = 0
    while index < len(pressed):
        table = runner.wrapper.current_action_table()
        context = runner.context()
        action = matching_action(table, pressed[index : index + table.shape[1]])
        step = 1 if action is None else table.shape[1]
        for buttons in pressed[index : index + step]:
            runner.press(buttons)
        if action is not None:
            yield context, action
        index += step


def learned_table(runner: SegmentRunner, routes: Routes) -> Table:
    counts: dict[str, Counter[int]] = defaultdict(Counter)
    for part in routes.parts:
        if part.name not in routes.routes or not routes.segments(part.name):
            continue
        pressed = np.concatenate([np.asarray(segment.pressed, dtype=np.int8) for segment in routes.segments(part.name)])
        for context, action in decisions(runner, Start(part.entry_state, routes.routes[part.name].entry), pressed):
            for key in dict.fromkeys(context):
                counts[key][action] += 1
    return {key: dict(counted) for key, counted in counts.items()}


def write_prior(runner: SegmentRunner, routes: Routes, path: Path) -> None:
    atomic_write(path, msgspec.msgpack.encode(learned_table(runner, routes)))


class ActionPrior:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.table: Table = {}
        self.loaded: int | None = None

    def counts(self, context: tuple[str, str]) -> dict[int, int] | None:
        changed = changed_at(self.path)
        if changed != self.loaded:
            self.table = {} if changed is None else msgspec.msgpack.decode(self.path.read_bytes(), type=Table)
            self.loaded = changed
        return next((self.table[key] for key in context if key in self.table), None)

    def choose(self, context: tuple[str, str], actions: int, rng: random.Random) -> int | None:
        counts = self.counts(context)
        allowed = {action: count for action, count in (counts or {}).items() if action < actions}
        if not allowed:
            return None
        return rng.choices(list(allowed), weights=list(allowed.values()))[0]
