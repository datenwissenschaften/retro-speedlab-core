from dataclasses import dataclass

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment.level_drill import LevelDrill


@dataclass(slots=True, frozen=True)
class Learning:
    curriculum: ReverseCurriculum
    drill: LevelDrill

    def __call__(self, state_name: str) -> bool:
        return self.drill.drills(state_name) or not self.curriculum.is_mastered(state_name)
