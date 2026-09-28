from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

FRAME = np.zeros((4, 4, 3), np.uint8)


@dataclass
class _FakeRam(RamInfo):
    value: int = 0


class _StateB(State):
    description = "Finish the level."


class _StateA(State):
    description = "Reach the exit."
    should_transition = False

    def _reward(self) -> float:
        return 1.0

    def _next(self):
        return _StateB if self.should_transition else None


def test_reset_without_a_state_type_returns_to_the_start_state(tmp_path: Path):
    machine = StateMachine(_StateA, Landmarks(tmp_path / "landmarks.json"))
    machine.current_state = _StateB(machine.landmarks)

    machine.reset(_FakeRam(), FRAME, None)

    assert machine.current_state is machine.start_state
    assert machine.last_transition is None


def test_reset_with_a_state_type_starts_in_that_state(tmp_path: Path):
    machine = StateMachine(_StateA, Landmarks(tmp_path / "landmarks.json"))

    machine.reset(_FakeRam(), FRAME, _StateB)

    assert machine.state_name == "_StateB"
    assert machine.question == "Finish the level."


def test_step_without_transition_keeps_the_state(tmp_path: Path):
    machine = StateMachine(_StateA, Landmarks(tmp_path / "landmarks.json"))
    machine.reset(_FakeRam(), FRAME, None)

    reward, terminated, truncated = machine.step(_FakeRam(), FRAME)

    assert (reward, terminated, truncated) == (1.0, False, False)
    assert machine.last_transition is None


def test_transition_switches_state_question_and_records_it(tmp_path: Path):
    machine = StateMachine(_StateA, Landmarks(tmp_path / "landmarks.json"))
    machine.reset(_FakeRam(), FRAME, None)
    machine.current_state.should_transition = True

    machine.step(_FakeRam(), FRAME)

    assert machine.last_transition == ("_StateA", "_StateB")
    assert machine.question == "Finish the level."


def test_states_are_reused_across_transitions(tmp_path: Path):
    machine = StateMachine(_StateA, Landmarks(tmp_path / "landmarks.json"))
    machine.reset(_FakeRam(), FRAME, _StateB)
    first = machine.current_state

    machine.reset(_FakeRam(), FRAME, _StateB)

    assert machine.current_state is first
