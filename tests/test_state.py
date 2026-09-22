from dataclasses import dataclass

import cv2
import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.state import State


@dataclass
class _FakeRam(RamInfo):
    position_x: int = 0
    position_y: int = 0


def _inputs():
    ram = _FakeRam()
    frame = np.zeros((8, 8), dtype=np.uint8)
    observation = np.zeros((1, 8, 8), dtype=np.uint8)
    return ram, frame, observation


def test_state_without_a_template_file_skips_target_detector_creation():
    state = State()

    assert not hasattr(state, "target_detector")


def test_state_skips_target_detector_when_the_template_asset_is_missing(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class _Templated(State):
        template_file = "missing.png"

    state = _Templated()

    assert not hasattr(state, "target_detector")


def test_state_creates_a_target_detector_when_the_template_asset_exists(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    cv2.imwrite(str(assets_dir / "template.png"), np.zeros((4, 4), dtype=np.uint8))

    class _Templated(State):
        template_file = "template.png"

    state = _Templated()

    assert state.target_detector.template_path.endswith("template.png")


def test_reset_stores_inputs_and_invokes_the_reset_hook():
    calls = []

    class _Tracking(State):
        def _on_reset(self):
            calls.append("reset")

    state = _Tracking()
    ram, frame, observation = _inputs()

    state.reset(ram, frame, observation)

    assert state.ram is ram
    assert state.frame is frame
    assert state.observation is observation
    assert calls == ["reset"]


def test_step_returns_the_reward_termination_and_next_state():
    class _Custom(State):
        def _reward(self):
            return 1.5

        def _terminated(self):
            return True

        def _truncated(self):
            return False

        def _next(self):
            return State

    state = _Custom()
    ram, frame, observation = _inputs()

    reward, terminated, truncated, next_state = state.step(ram, frame, observation)

    assert (reward, terminated, truncated, next_state) == (1.5, True, False, State)
    assert state.ram is ram


def test_features_combines_the_visual_encoding_with_auxiliary_features():
    class _Aux(State):
        def auxiliary_features(self, ram):
            return [9.0]

    state = _Aux()
    _, _, observation = _inputs()
    state.ram = _FakeRam()
    state.observation = observation

    features = state.features()

    assert features[-1] == 9.0
    assert len(features) > 1


def test_default_hooks_report_baseline_values():
    state = State()

    assert state._reward() == 0.0
    assert state._terminated() is False
    assert state._truncated() is False
    assert state._won() is False
    assert state._next() is None
    assert state.auxiliary_features() == []
