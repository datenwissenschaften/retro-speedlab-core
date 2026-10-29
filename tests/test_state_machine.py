from dataclasses import dataclass

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State


@dataclass
class _FakeRam(RamInfo):
    value: int = 0


class _StateB(State):
    goal = "Finish the level"


class _StateA(State):
    goal = "Reach the exit"
    should_transition = False

    def _reward(self) -> float:
        return 1.0

    def _next(self):
        return _StateB if self.should_transition else None


def test_reset_without_a_state_type_returns_to_the_start_state():
    machine = StateMachine(_StateA)
    machine.current_state = _StateB()

    machine.reset(_FakeRam(), None)

    assert machine.current_state is machine.start_state
    assert machine.last_transition is None


def test_reset_with_a_state_type_starts_in_that_state():
    machine = StateMachine(_StateA)

    machine.reset(_FakeRam(), _StateB)

    assert machine.state_name == "_StateB"


def test_step_without_transition_keeps_the_state():
    machine = StateMachine(_StateA)
    machine.reset(_FakeRam(), None)

    reward, terminated, truncated = machine.step(_FakeRam())

    assert (reward, terminated, truncated) == (1.0, False, False)
    assert machine.last_transition is None


def test_transition_switches_state_and_records_it():
    machine = StateMachine(_StateA)
    machine.reset(_FakeRam(), None)
    machine.current_state.should_transition = True

    machine.step(_FakeRam())

    assert machine.last_transition == ("_StateA", "_StateB")
    assert machine.state_name == "_StateB"


def test_states_are_reused_across_transitions():
    machine = StateMachine(_StateA)
    machine.reset(_FakeRam(), _StateB)
    first = machine.current_state

    machine.reset(_FakeRam(), _StateB)

    assert machine.current_state is first
