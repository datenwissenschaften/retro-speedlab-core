import math
import random
import time
from dataclasses import dataclass

from datenwissenschaften.tas.exploration import Exploration, Step, step_of
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.search import choose
from datenwissenschaften.tas.strategies import actions_of, next_action

TREE_SECONDS = 20.0
TREE_SIMULATIONS = 256
ROLLOUT_DECISIONS = 8
UCB_WEIGHT = 1.4
NEW_PLACE_VALUE = 1.0


@dataclass(slots=True)
class Node:
    start: Start
    step: Step
    visits: int
    value: float
    children: dict[int, "Node | None"]


def ucb(parent: Node, child: Node) -> float:
    return child.value / child.visits + UCB_WEIGHT * math.sqrt(math.log(parent.visits) / child.visits)


def selected(root: Node, actions: int) -> list[Node] | None:
    path = [root]
    while len(path[-1].children) == actions:
        alive = [child for child in path[-1].children.values() if child is not None]
        if not alive:
            return None
        path.append(max(alive, key=lambda child: ucb(path[-1], child)))
    return path


def rollout(exploration: Exploration, step: Step, rng: random.Random) -> float:
    found, action = 0, rng.randrange(actions_of(exploration))
    for _ in range(ROLLOUT_DECISIONS):
        action = next_action(exploration, rng, action)
        step = exploration.act(step, action)
        if step.ended:
            break
        found += exploration.remember(step)
    return step.score + NEW_PLACE_VALUE * found


def simulate(exploration: Exploration, root: Node, rng: random.Random) -> bool:
    actions = actions_of(exploration)
    path = selected(root, actions)
    if path is None:
        return False
    leaf = path[-1]
    action = rng.choice([action for action in range(actions) if action not in leaf.children])
    exploration.runner.restore(leaf.start)
    step = exploration.act(leaf.step, action)
    if step.ended:
        leaf.children[action] = None
        return True
    found = exploration.remember(step)
    child = Node(Start(leaf.start.state, exploration.runner.snapshot()), step, 0, 0.0, {})
    leaf.children[action] = child
    value = rollout(exploration, step, rng) + NEW_PLACE_VALUE * found - root.step.score
    for node in (*path, child):
        node.visits += 1
        node.value += value
    return True


def tree(exploration: Exploration, rng: random.Random) -> None:
    cell = choose(exploration.cells(), rng)
    root = Node(cell.start, step_of(cell), 0, 0.0, {})
    budget = min(exploration.deadline, time.monotonic() + TREE_SECONDS)
    for _ in range(TREE_SIMULATIONS):
        if time.monotonic() >= budget or not simulate(exploration, root, rng):
            return
