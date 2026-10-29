import multiprocessing
import random
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from functools import cache
from pathlib import Path

from loguru import logger

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.imitation import Example
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.workers import teach
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.lessons import Lessons

SNAPSHOT_DECISIONS = 16
SNAPSHOTS = 4
TEACH_SECONDS = 60.0


@cache
def teacher_pool() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))


class ExpertHook:
    def __init__(self, env: StateMachineGymWrapper, lessons: Lessons, config: Path) -> None:
        self.env = env
        self.lessons = lessons
        self.wrapper = f"{type(env).__module__}:{type(env).__qualname__}"
        self.config = config
        self.snapshots: deque[Start] = deque(maxlen=SNAPSHOTS)
        self.steps = 0
        self.lesson: tuple[str, Future[list[Example]]] | None = None

    def on_step(self, transition: Transition) -> None:
        state = transition.info["state"]
        if self.snapshots and self.snapshots[-1].state != state:
            self.snapshots.clear()
        self.steps += 1
        if self.lessons.learning(state) and self.steps % SNAPSHOT_DECISIONS == 0:
            self.snapshots.append(Start(state, bytes(self.env.unwrapped.em.get_state())))
        if self.lesson is not None and self.lesson[1].done():
            state, examples = self.lesson[0], self.lesson[1].result()
            self.lesson = None
            if examples:
                self.lessons.wins.keep(state, examples)
                logger.info(f"The search taught Laya {len(examples)} decisions out of a failed attempt in {state}")

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        if episode.curriculum_succeeded or self.lesson is not None or not self.snapshots:
            return
        start = self.snapshots[0]
        lesson = teacher_pool().submit(teach, self.wrapper, self.config, start, random.randrange(2**31), TEACH_SECONDS)
        self.lesson = (start.state, lesson)
        self.snapshots.clear()

    def on_update(self) -> None:
        pass
