from loguru import logger

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment.level_clock import LevelClock
from datenwissenschaften.environment.level_drill import LevelDrill
from datenwissenschaften.environment.levels import LevelTargets


class CurriculumOutcomes:
    def __init__(self, targets: LevelTargets, drill: LevelDrill, clock: LevelClock) -> None:
        self.targets = targets
        self.drill = drill
        self.clock = clock

    def succeed(self, curriculum: ReverseCurriculum, target: str, steps: int) -> bool:
        if self.drill.level is not None:
            self.drill.finish(target, True)
            return False
        if self.targets.is_level(target):
            self.clock.record(target, steps)
        mastered = curriculum.record_success(target, steps)
        if mastered:
            logger.info(f"Mastered {target} in {steps} steps")
        return mastered

    def fail(self, curriculum: ReverseCurriculum, target: str, steps: int, score: float, ending: str) -> None:
        if self.drill.level is not None:
            self.drill.finish(target, False)
            return
        if self.targets.is_level(target):
            self.drill.record_ending(target, ending)
        if curriculum.record_failure(target, steps, score):
            logger.warning(f"Deleted score-stagnant automatic checkpoint for {target}")
