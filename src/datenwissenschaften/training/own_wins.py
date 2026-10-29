from collections.abc import Callable
from pathlib import Path

import msgspec

from datenwissenschaften.curriculum import atomic_write
from datenwissenschaften.environment.observer import Observation
from datenwissenschaften.laya.imitation import Example

WIN_EXAMPLES = 2048


class OwnWins:
    def __init__(self, order: tuple[str, ...], path: Callable[[str], Path], learning: Callable[[str], bool]) -> None:
        self.order = order
        self.path = path
        self.learning = learning
        self.attempts: dict[int, list[Example]] = {}
        self.wins: dict[str, list[Example]] = {}

    def examples(self, state: str) -> list[Example]:
        if state not in self.wins:
            saved = self.path(state)
            self.wins[state] = msgspec.json.decode(saved.read_bytes(), type=list[Example]) if saved.is_file() else []
        return self.wins[state]

    def add(
        self, environment: int, state: str, observation: Observation, action: int, reached: str, ended: bool
    ) -> None:
        if not self.learning(state):
            if environment in self.attempts:
                del self.attempts[environment]
            return
        attempt = self.attempts.setdefault(environment, [])
        attempt.append(Example(observation["state"], observation["question"], action))
        if reached == state and not ended:
            return
        del self.attempts[environment]
        if reached in self.order and self.order.index(reached) > self.order.index(state):
            self.keep(state, attempt)

    def keep(self, state: str, examples: list[Example]) -> None:
        self.wins[state] = [*self.examples(state), *examples][-WIN_EXAMPLES:]
        atomic_write(self.path(state), msgspec.json.encode(self.wins[state]))
