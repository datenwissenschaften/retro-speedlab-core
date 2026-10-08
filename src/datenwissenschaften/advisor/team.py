from collections.abc import Callable

from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.advisor.backplay import Backplay
from datenwissenschaften.advisor.coach import Coach
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.practice import PracticeEnvironments


class AdvisorTeam:
    def __init__(
        self,
        advisors: Advisors,
        practice: PracticeEnvironments,
        lab_run: LabRun,
        lessons: Callable[[str], list[DemonstrationStep]],
        backplay: Backplay,
    ) -> None:
        self.advisors = advisors
        self.practice = practice
        self.coach = Coach(advisors, practice, lab_run, lessons, backplay)

    def close(self) -> None:
        self.coach.stop()
        self.practice.close()
        self.advisors.close()
