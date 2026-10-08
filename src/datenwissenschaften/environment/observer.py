from collections.abc import Callable
from typing import Generic, TypeVar

import numpy as np

from datenwissenschaften.advisor.advice import Advice
from datenwissenschaften.advisor.inputs import encode
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.facts import Facts, render_facts
from datenwissenschaften.states.machine import StateMachine

T = TypeVar("T", bound=RamInfo)

Observation = dict[str, str]
Advise = Callable[[str, np.ndarray], Advice]
ADVISOR_FACT = "advised"


class Observer(Generic[T]):
    def __init__(
        self, read_bytes: Callable[[], np.ndarray], state_machine: StateMachine[T], action_descriptions: tuple[str, ...]
    ) -> None:
        self.read_bytes = read_bytes
        self.action_descriptions = action_descriptions
        self.state_machine = state_machine
        self.advisor: Advise | None = None
        self.advice: Advice | None = None

    def facts(self, ram: T) -> Facts:
        return {**ram.describe(), **self.state_machine.current_state.describe()}

    def inputs(self, ram: T) -> np.ndarray:
        return encode(self.read_bytes(), self.facts(ram))

    def observation(self, ram: T) -> Observation:
        facts = self.facts(ram)
        if self.advisor is not None:
            self.advice = self.advisor(self.state_machine.state_name, encode(self.read_bytes(), facts))
            facts = {ADVISOR_FACT: self.action_descriptions[self.advice.action], **facts}
        return {"state": render_facts(facts), "question": self.state_machine.question}

    def advised_action(self) -> int | None:
        return None if self.advice is None else self.advice.action
