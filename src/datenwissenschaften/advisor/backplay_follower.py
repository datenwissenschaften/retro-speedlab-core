from datenwissenschaften.advisor.backplay import Backplay
from datenwissenschaften.training.practice import PracticeEnvironments, PracticeStep


class BackplayFollower:
    def __init__(self, backplay: Backplay, practice: PracticeEnvironments) -> None:
        self.backplay = backplay
        self.practice = practice
        self.backplayed: dict[int, str] = {}

    def follow(self, worker: int, step: PracticeStep) -> None:
        if worker in self.backplayed:
            origin = self.backplayed[worker]
            if self.backplay.forward(origin, step.reached) or step.ended:
                self.backplay.record(origin, self.backplay.forward(origin, step.reached))
                del self.backplayed[worker]
        if not step.ended or (start := self.backplay.start(step.state)) is None:
            return
        self.practice.restart(worker, step.state, start)
        self.backplayed[worker] = step.state
