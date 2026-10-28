from dataclasses import dataclass


@dataclass(slots=True)
class EpisodeTally:
    steps: int
    segment_return: float
    score: float

    def count_step(self) -> None:
        self.steps += 1

    def add_reward(self, reward: float, transitioned: bool) -> None:
        self.score += reward
        self.segment_return = 0.0 if transitioned else self.segment_return + reward
