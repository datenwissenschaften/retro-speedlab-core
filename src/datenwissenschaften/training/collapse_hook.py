from collections import Counter

from loguru import logger

from datenwissenschaften.environment.level_drill import LevelDrill
from datenwissenschaften.ppo import ENTROPY_FLOOR
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels

COLLAPSED_UPDATES = 50


class CollapseHook:
    def __init__(self, models: StateModels, drill: LevelDrill) -> None:
        self.models = models
        self.drill = drill
        self.collapsed: Counter[str] = Counter()
        self.restarted: set[str] = set()

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        state = self.models.require_active()
        if not self.drill.drills(state) or self.models.agent.entropy_share >= ENTROPY_FLOOR:
            del self.collapsed[state]
            return
        self.collapsed[state] += 1
        if self.collapsed[state] < COLLAPSED_UPDATES or state in self.restarted:
            return
        self.models.reset(state)
        self.restarted.add(state)
        logger.warning(
            f"Laya's {state} head stayed collapsed for {COLLAPSED_UPDATES} drill updates; "
            "it starts again from Laya's own judgement and the lessons"
        )
