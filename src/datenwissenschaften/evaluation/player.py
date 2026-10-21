from dataclasses import dataclass

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.training.state_models import StateModels

GREEDY = 0.0


@dataclass(slots=True, frozen=True)
class Attempt:
    score: float
    progress: float
    won: bool


class Player:
    def __init__(self, env: StateMachineGymWrapper, models: StateModels, phases: tuple[str, ...]) -> None:
        self.env = env
        self.models = models
        self.phases = phases

    def play(self, start: str | None, decisions: int) -> Attempt:
        observation, info = self.env.reset_to(start)
        score, progress, won = 0.0, self.phases.index(info["state"]), False
        for _ in range(decisions):
            state = info["state"]
            decision = self.models.decide([observation], [state], {state: GREEDY}, [info["advice"]])[0]
            observation, reward, terminated, truncated, info = self.env.step(decision.action)
            score += reward
            progress = max(progress, self.phases.index(info["state"]))
            won = won or info["won"]
            if terminated or truncated:
                break
        return Attempt(score, float(progress), won)
