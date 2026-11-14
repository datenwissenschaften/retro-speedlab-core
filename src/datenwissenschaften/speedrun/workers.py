from dataclasses import replace
from functools import cache
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.wrapper import state_class
from datenwissenschaften.knowledge import knowledge_path, load_knowledge
from datenwissenschaften.policy.prior import POLICY_FILE, PolicyPrior
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.archives import Archives
from datenwissenschaften.speedrun.beam import BEAM_MODE, beam_search
from datenwissenschaften.speedrun.explore import search_segment
from datenwissenschaften.speedrun.live import BEAM_SCOUT, VISION_SCOUT, WorkerLive
from datenwissenschaften.speedrun.paths import archives_dir, tas_dir
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.prior import PRIOR_FILE, ActionPrior
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing, Searched, shave
from datenwissenschaften.speedrun.strategies import WALK
from datenwissenschaften.speedrun.vision import VISION_MODE, vision_search
from datenwissenschaften.workflow.wrapper import wrapper_class


@cache
def worker_runner(wrapper: str, config: Path) -> SegmentRunner:
    return SegmentRunner(make_environment(wrapper_class(wrapper), load_config(config)))


@cache
def worker_preferences(config: Path) -> Preferences:
    output = tas_dir(load_config(config))
    return Preferences(ActionPrior(output / PRIOR_FILE), PolicyPrior(output / POLICY_FILE))


@cache
def worker_archives(config: Path) -> Archives:
    settings = load_config(config)
    return Archives(archives_dir(settings), settings.speedrun.scouts + settings.speedrun.polishers)


def explore(
    wrapper: str, config: Path, start: Start, bound: int, seed: int, seconds: float, live: Path | None, role: str
) -> Searched:
    watched = None if live is None else WorkerLive(live, seed, role, WALK)
    runner, preferences = worker_runner(wrapper, config), worker_preferences(config)
    return search_segment(runner, start, bound, seed, seconds, watched, preferences, worker_archives(config))


def polish(wrapper: str, config: Path, start: Start, crossing: Crossing, seconds: float) -> Crossing:
    return shave(worker_runner(wrapper, config), start, crossing, seconds)


def beam(wrapper: str, config: Path, start: Start, seed: int, seconds: float, live: Path | None) -> Searched:
    watched = None if live is None else WorkerLive(live, seed, BEAM_SCOUT, BEAM_MODE)
    return beam_search(worker_runner(wrapper, config), start, seconds, watched)


def vision(wrapper: str, config: Path, start: Start, seed: int, live: Path | None) -> Searched:
    runner = worker_runner(wrapper, config)
    watched = None if live is None else WorkerLive(live, seed, VISION_SCOUT, VISION_MODE)
    goal = state_class(runner.wrapper.state_classes, start.state).goal
    return vision_search(runner, start, goal, worker_notes(config), watched)


@cache
def worker_notes(config: Path) -> str:
    knowledge = load_knowledge(knowledge_path(load_config(config).paths.game_dir))
    return "\n".join(f"- {mechanic.title}: {mechanic.text}" for mechanic in knowledge.mechanics)


def explore_tail(
    wrapper: str,
    config: Path,
    start: Start,
    head: list[list[int]],
    bound: int,
    seed: int,
    seconds: float,
    live: Path | None,
    role: str,
) -> Searched:
    runner = worker_runner(wrapper, config)
    runner.restore(start)
    for buttons in head:
        runner.press(np.asarray(buttons, dtype=np.int8))
    cut = runner.here(start.state)
    searched = explore(wrapper, config, cut, bound, seed, seconds, live, role)
    if not head:
        return searched
    played = np.asarray(head, dtype=np.int8)
    return Searched(
        [replace(crossing, pressed=np.concatenate([played, crossing.pressed])) for crossing in searched.crossings],
        None,
    )
