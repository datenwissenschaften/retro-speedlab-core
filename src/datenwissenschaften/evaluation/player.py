from dataclasses import dataclass

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.training.state_models import StateModels

GREEDY = 0.0


@dataclass(slots=True, frozen=True)
class Attempt:
    progress: float
    won: bool
    decisions: int


class Player:
    def __init__(self, env: StateMachineGymWrapper, models: StateModels, phases: tuple[str, ...]) -> None:
        self.env = env
        self.models = models
        self.phases = phases

    def play(self, decisions: int) -> Attempt:
        observation, info = self.env.reset_to(None)
        furthest, won, taken = self.phases.index(info["state"]), False, 0
        while taken < decisions:
            state = info["state"]
            decision = self.models.decide([observation], [state], {state: GREEDY}, [info["advice"]])[0]
            observation, reward, terminated, truncated, info = self.env.step(decision.action)
            taken += 1
            furthest = max(furthest, self.phases.index(info["state"]))
            won = won or info["won"]
            if terminated or truncated:
                break
        return Attempt(furthest / max(len(self.phases) - 1, 1), won, taken)
