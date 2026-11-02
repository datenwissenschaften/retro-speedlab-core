import random
import time
from collections import deque
from itertools import product

from datenwissenschaften.tas.exploration import Exploration, Step, step_of
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.search import EXPLORE_DECISIONS, KEEP_ACTION, STILL_FRAMES, choose

PRIOR_SHARE = 0.3
FRONTIER_DEPTH = 3
FRONTIER_HOLDS = (1, 3, 8)
FRONTIER_SECONDS = 20.0


def actions_of(exploration: Exploration) -> int:
    return len(exploration.runner.wrapper.action_table)


def next_action(exploration: Exploration, rng: random.Random, action: int) -> int:
    actions = actions_of(exploration)
    if rng.random() < PRIOR_SHARE:
        preferred = exploration.prior.choose(exploration.runner.context(), actions, rng)
        if preferred is not None:
            return preferred
    return action if rng.random() < KEEP_ACTION else rng.randrange(actions)


def walk(exploration: Exploration, rng: random.Random) -> None:
    cell = choose(list(exploration.archive.values()), rng)
    exploration.runner.restore(cell.start)
    step, action = step_of(cell), rng.randrange(actions_of(exploration))
    last, decisions, still = exploration.runner.key(step.frames), 0, 0
    while decisions < EXPLORE_DECISIONS and still < STILL_FRAMES:
        action = next_action(exploration, rng, action)
        before, step = step.frames, exploration.act(step, action)
        if step.ended:
            return
        key = exploration.runner.key(step.frames)
        if key == last:
            still += step.frames - before
            continue
        last, decisions, still = key, decisions + 1, 0
        exploration.remember(step)


def held(exploration: Exploration, step: Step, action: int, hold: int) -> Step:
    for _ in range(hold):
        step = exploration.act(step, action)
        if step.ended:
            break
    return step


def frontier(exploration: Exploration, rng: random.Random) -> None:
    cell = choose(list(exploration.archive.values()), rng)
    budget = min(exploration.deadline, time.monotonic() + FRONTIER_SECONDS)
    layer = deque([(cell.start, step_of(cell), FRONTIER_DEPTH)])
    while layer:
        start, step, depth = layer.popleft()
        for action, hold in product(range(actions_of(exploration)), FRONTIER_HOLDS):
            if time.monotonic() >= budget:
                return
            exploration.runner.restore(start)
            reached = held(exploration, step, action, hold)
            if reached.ended:
                continue
            exploration.remember(reached)
            if depth > 1:
                layer.append((Start(start.state, exploration.runner.snapshot()), reached, depth - 1))
