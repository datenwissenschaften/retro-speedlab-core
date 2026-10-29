import os
from functools import cache
from pathlib import Path

from datenwissenschaften.environment.demonstration import nearest_action
from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.laya.imitation import Example
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import UNBOUNDED, Crossing, search_segment, shave
from datenwissenschaften.workflow.wrapper import wrapper_class


@cache
def worker_runner(wrapper: str, config: Path) -> SegmentRunner:
    return SegmentRunner(make_environment(wrapper_class(wrapper), load_config(config), os.getpid(), False))


def explore(wrapper: str, config: Path, start: Start, bound: int, seed: int, seconds: float) -> list[Crossing]:
    return search_segment(worker_runner(wrapper, config), start, bound, seed, seconds)


def polish(wrapper: str, config: Path, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    return shave(worker_runner(wrapper, config), start, crossing, seconds)


def teach(wrapper: str, config: Path, start: Start, seed: int, seconds: float) -> list[Example]:
    runner = worker_runner(wrapper, config)
    crossings = search_segment(runner, start, UNBOUNDED, seed, seconds)
    if not crossings:
        return []
    runner.restore(start)
    table, examples = runner.wrapper.current_action_table(), []
    frames_per_decision = table.shape[1]
    pressed = crossings[0].pressed
    for first in range(0, len(pressed) - frames_per_decision + 1, frames_per_decision):
        chunk = pressed[first : first + frames_per_decision]
        observation = runner.wrapper.observer.observation(runner.ram)
        examples.append(Example(observation["state"], observation["question"], nearest_action(table, chunk)))
        for buttons in chunk:
            runner.press(buttons)
    return examples
