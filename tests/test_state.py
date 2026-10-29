from dataclasses import dataclass

import pytest

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State


@dataclass
class _FakeRam(RamInfo):
    value: int = 0


class _Idle(State):
    goal = "Wait"


class _Aimless(State):
    pass


class _Counting(State):
    goal = "Count"

    def _on_reset(self) -> None:
        self.resets = 1

    def _reward(self) -> float:
        return float(self.ram.value)

    def _terminated(self) -> bool:
        return self.ram.value > 1


def test_default_state_is_neutral():
    state = _Idle()
    state.reset(_FakeRam())

    assert state.step(_FakeRam()) == (0.0, False, False, None)
    assert state._won() is False
    assert state.goal == "Wait"


def test_a_state_without_a_goal_fails_when_it_is_read():
    with pytest.raises(AttributeError):
        _Aimless().goal


def test_state_hooks_see_the_latest_ram():
    state = _Counting()
    state.reset(_FakeRam())

    assert state.resets == 1
    assert state.step(_FakeRam(value=2)) == (2.0, True, False, None)
