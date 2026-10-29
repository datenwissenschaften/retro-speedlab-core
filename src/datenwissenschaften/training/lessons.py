from dataclasses import dataclass

from datenwissenschaften.environment.demonstration import Demonstrations
from datenwissenschaften.laya.imitation import (
    IMITATION_WEIGHT,
    MASTERED_IMITATION_WEIGHT,
    DemonstrationStep,
    Lesson,
    Shown,
)
from datenwissenschaften.training.learning import Learning
from datenwissenschaften.training.own_wins import OwnWins


@dataclass(slots=True, frozen=True)
class Lessons:
    demonstrations: Demonstrations
    learning: Learning
    wins: OwnWins

    def __call__(self, state_name: str) -> Lesson[DemonstrationStep]:
        return Lesson(self._shown(state_name), self._weight(state_name))

    def laya(self, state_name: str) -> Lesson[Shown]:
        return Lesson([*self._shown(state_name), *self.wins.examples(state_name)], self._weight(state_name))

    def _shown(self, state_name: str) -> list[DemonstrationStep]:
        return self.demonstrations[state_name] if state_name in self.demonstrations else []

    def _weight(self, state_name: str) -> float:
        return IMITATION_WEIGHT if self.learning(state_name) else MASTERED_IMITATION_WEIGHT
