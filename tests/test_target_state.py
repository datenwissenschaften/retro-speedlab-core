from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states import target as target_module
from datenwissenschaften.states.target import TargetState
from datenwissenschaften.states.target_memory import TargetMemory


@dataclass
class _FakeRam(RamInfo):
    position_x: int = 0
    position_y: int = 0
    screen_x: int = 0
    screen_y: int = 0


class _FakeDetector:
    def __init__(self, template_file):
        self.template_file = template_file
        self.seen = False
        self.position = None
        self.detected_frames = []

    def detect(self, frame):
        self.detected_frames.append(frame)

    def distance(self, position):
        return 42.0 if self.seen else None


class _FakeMemory:
    def __init__(self):
        self.remembered = []
        self.features_calls = []

    def features(self, coordinates):
        self.features_calls.append(coordinates)
        return [1.0, 0.0]

    def remember(self, coordinates):
        self.remembered.append(coordinates)
        return True


class _ConcreteTarget(TargetState[_FakeRam]):
    template_file = "unused.png"


def _target(monkeypatch):
    monkeypatch.setattr(target_module, "TemplateDetector", lambda template_file: _FakeDetector(template_file))
    monkeypatch.setattr(TargetMemory, "shared", classmethod(lambda cls, key, *, origin, scale: _FakeMemory()))
    return _ConcreteTarget()


def test_init_wires_the_template_detector_and_target_memory(monkeypatch):
    target = _target(monkeypatch)

    assert isinstance(target.target_detector, _FakeDetector)
    assert isinstance(target.target_memory, _FakeMemory)
    # State.__init__'s own template-file check must not create a second detector.
    assert target.target_detector.template_file == "unused.png"


def test_auxiliary_features_reports_unknown_target_without_ram(monkeypatch):
    target = _target(monkeypatch)

    features = target.auxiliary_features(None)

    assert features == [1.0, 0.0]
    assert target.target_memory.features_calls == [None]


def test_auxiliary_features_passes_the_actor_position_when_ram_is_known(monkeypatch):
    target = _target(monkeypatch)
    ram = _FakeRam(position_x=10, position_y=20, screen_x=1, screen_y=2)

    target.auxiliary_features(ram)

    assert target.target_memory.features_calls == [(266, 532)]


def test_remember_detected_target_is_a_no_op_when_nothing_is_seen(monkeypatch):
    target = _target(monkeypatch)
    target.ram = _FakeRam()

    assert target.remember_detected_target() is False
    assert target.target_memory.remembered == []


def test_remember_detected_target_stores_the_detected_screen_position(monkeypatch):
    target = _target(monkeypatch)
    target.ram = _FakeRam(screen_x=1, screen_y=1)
    target.target_detector.seen = True
    target.target_detector.position = Position(5, 5)

    assert target.remember_detected_target() is True
    assert target.target_memory.remembered == [(261, 261)]


def test_on_reset_clears_the_missing_target_streak(monkeypatch):
    target = _target(monkeypatch)
    target.target_missing_steps = 7

    target._on_reset()

    assert target.target_missing_steps == 0


def test_reward_penalizes_a_missing_target(monkeypatch):
    target = _target(monkeypatch)
    target.ram = _FakeRam()
    target.frame = np.zeros((4, 4), dtype=np.uint8)
    target.target_missing_steps = 0
    target.target_detector.seen = False

    reward = target._reward()

    assert reward == -(target.step_penalty + target.target_missing_penalty)
    assert target.target_missing_steps == 1
    assert target.target_detector.detected_frames == [target.frame]


def test_reward_gives_a_proximity_bonus_when_the_target_is_close(monkeypatch):
    target = _target(monkeypatch)
    target.ram = _FakeRam()
    target.frame = np.zeros((4, 4), dtype=np.uint8)
    target.target_missing_steps = 3
    target.target_detector.seen = True

    reward = target._reward()

    expected = -target.step_penalty + (target.stay_near_distance - 42.0) * target.proximity_reward_scale
    assert reward == pytest.approx(expected)
    assert target.target_missing_steps == 0


def test_target_memory_key_uses_the_template_file_stem(monkeypatch):
    target = _target(monkeypatch)

    assert target._target_memory_key() == Path("unused.png").stem


def test_actor_position_defaults_missing_ram_fields_to_zero(monkeypatch):
    target = _target(monkeypatch)

    position = target._actor_position(object())

    assert position.coordinates == (0, 0)
    assert position.screen == (0, 0)
