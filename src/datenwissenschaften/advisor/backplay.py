import random

from loguru import logger

from datenwissenschaften.environment.demonstration import StartPoints

SUCCESS_WINDOW = 20
ADVANCE_RATE = 0.5


class Backplay:
    def __init__(self, starts: StartPoints, order: tuple[str, ...]) -> None:
        self.starts = {state: points for state, points in starts.items() if points}
        self.order = order
        self.depth = dict.fromkeys(self.starts, 1)
        self.outcomes: dict[str, list[bool]] = {state: [] for state in self.starts}
        self.random = random.Random()

    def start(self, state: str) -> bytes | None:
        if state not in self.starts or self.finished(state):
            return None
        return self.random.choice(self.starts[state][-self.depth[state] :])

    def finished(self, state: str) -> bool:
        return self.depth[state] > len(self.starts[state])

    def forward(self, state: str, reached: str) -> bool:
        return reached in self.order and self.order.index(reached) > self.order.index(state)

    def record(self, state: str, succeeded: bool) -> None:
        outcomes = self.outcomes[state]
        outcomes.append(succeeded)
        if len(outcomes) < SUCCESS_WINDOW:
            return
        if sum(outcomes) / len(outcomes) >= ADVANCE_RATE:
            self.depth[state] += 1
            logger.info(f"Backplay for {state} starts {self.progress(state)} start points before its exit")
        outcomes.clear()

    def progress(self, state: str) -> str:
        return f"{min(self.depth[state], len(self.starts[state]))}/{len(self.starts[state])}"
