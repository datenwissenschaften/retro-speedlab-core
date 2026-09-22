from dataclasses import dataclass

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State


@dataclass
class _FakeRam(RamInfo):
    value: int = 0


class _StateB(State):
    pass


class _StateA(State):
    should_transition = False

    def _next(self):
        return _StateB if self.should_transition else None


def _inputs():
    ram = _FakeRam()
    frame = np.zeros((8, 8), dtype=np.uint8)
    observation = np.zeros((1, 8, 8), dtype=np.uint8)
    return ram, frame, observation


def _machine(*, terminate_on_transition, transition_bonus, on_transition):
    return StateMachine(
        _StateA(),
        terminate_on_transition=terminate_on_transition,
        transition_bonus=transition_bonus,
        on_transition=on_transition,
    )


def test_reset_without_a_state_type_returns_to_the_start_state():
    machine = _machine(terminate_on_transition=False, transition_bonus=0.0, on_transition=lambda *_: None)
    ram, frame, observation = _inputs()
    machine.current_state = _StateB()

    machine.reset(ram, frame, observation)

    assert machine.current_state is machine.start_state
    assert machine.last_transition is None


def test_reset_with_a_state_type_creates_and_caches_that_state():
    machine = _machine(terminate_on_transition=False, transition_bonus=0.0, on_transition=lambda *_: None)
    ram, frame, observation = _inputs()

    machine.reset(ram, frame, observation, _StateB)

    assert isinstance(machine.current_state, _StateB)
    assert _StateB in machine.states_by_type


def test_step_without_a_transition_keeps_the_current_state():
    machine = _machine(terminate_on_transition=True, transition_bonus=5.0, on_transition=lambda *_: None)
    ram, frame, observation = _inputs()
    machine.reset(ram, frame, observation)

    reward, terminated, truncated = machine.step(ram, frame, observation)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert machine.last_transition is None
    assert machine.current_state is machine.start_state


def test_step_with_a_transition_applies_the_bonus_and_notifies_listeners():
    notifications = []
    machine = _machine(
        terminate_on_transition=False,
        transition_bonus=5.0,
        on_transition=lambda before, after: notifications.append((before, after)),
    )
    ram, frame, observation = _inputs()
    machine.reset(ram, frame, observation)
    machine.current_state.should_transition = True

    reward, terminated, truncated = machine.step(ram, frame, observation)

    assert reward == 5.0
    assert terminated is False
    assert machine.last_transition == ("_StateA", "_StateB")
    assert notifications == [("_StateA", "_StateB")]
    assert isinstance(machine.current_state, _StateB)


def test_transitions_can_be_configured_to_end_the_episode():
    machine = _machine(terminate_on_transition=True, transition_bonus=0.0, on_transition=lambda *_: None)
    ram, frame, observation = _inputs()
    machine.reset(ram, frame, observation)
    machine.current_state.should_transition = True

    _, terminated, _ = machine.step(ram, frame, observation)

    assert terminated is True


def test_features_delegates_to_the_current_state():
    machine = _machine(terminate_on_transition=False, transition_bonus=0.0, on_transition=lambda *_: None)
    ram, frame, observation = _inputs()
    machine.reset(ram, frame, observation)

    assert machine.features() == machine.current_state.features()


def test_state_name_reports_the_current_state_class_name():
    machine = _machine(terminate_on_transition=False, transition_bonus=0.0, on_transition=lambda *_: None)

    assert machine.state_name == "_StateA"
