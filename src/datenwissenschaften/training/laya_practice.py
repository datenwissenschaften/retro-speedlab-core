from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.environment.observer import Observation, observe
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.training.practice import PracticeEnvironments, PracticeStep


class LayaPractice:
    def __init__(self, practice: PracticeEnvironments, advisors: Advisors, actions: tuple[str, ...]) -> None:
        self.practice = practice
        self.advisors = advisors
        self.actions = actions

    @property
    def states(self) -> list[str]:
        return self.practice.states

    def views(self) -> tuple[list[Observation], list[int]]:
        views = zip(self.practice.states, self.practice.views, strict=True)
        advice = [(view, self.advisors.advise(state, view.inputs).action) for state, view in views]
        observations = [observe(view.facts, view.question, self.actions[action]) for view, action in advice]
        return observations, [action for _, action in advice]

    def send(self, decisions: list[Decision]) -> None:
        self.practice.send([decision.action for decision in decisions])

    def receive(self) -> list[PracticeStep]:
        return self.practice.receive()

    def close(self) -> None:
        self.practice.close()
