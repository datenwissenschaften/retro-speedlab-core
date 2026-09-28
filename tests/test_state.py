from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.states.state import State

FRAME = np.zeros((4, 4, 3), np.uint8)


@dataclass
class _FakeRam(RamInfo):
    value: int = 0


class _Idle(State):
    description = "Wait."


class _Counting(State):
    description = "Count."

    def _on_reset(self) -> None:
        self.resets = 1

    def _reward(self) -> float:
        return float(self.ram.value)

    def _terminated(self) -> bool:
        return self.ram.value > 1


def test_default_state_is_neutral(tmp_path: Path):
    state = _Idle(Landmarks(tmp_path / "landmarks.json"))
    state.reset(_FakeRam(), FRAME)

    assert state.step(_FakeRam(), FRAME) == (0.0, False, False, None)
    assert state._won() is False
    assert state.describe() == {}
    assert state.detections() == ()


def test_state_hooks_see_the_latest_ram(tmp_path: Path):
    state = _Counting(Landmarks(tmp_path / "landmarks.json"))
    state.reset(_FakeRam(), FRAME)

    assert state.resets == 1
    assert state.step(_FakeRam(value=2), FRAME) == (2.0, True, False, None)
