from dataclasses import dataclass

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment.demonstration import Demonstrations
from datenwissenschaften.laya.imitation import DemonstrationStep


@dataclass(slots=True, frozen=True)
class Lessons:
    demonstrations: Demonstrations
    curriculum: ReverseCurriculum

    def __call__(self, state_name: str) -> list[DemonstrationStep]:
        if state_name not in self.demonstrations or self.curriculum.is_mastered(state_name):
            return []
        return self.demonstrations[state_name]
