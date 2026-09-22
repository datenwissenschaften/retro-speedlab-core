from dataclasses import dataclass

import pytest

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states import target as target_module
from datenwissenschaften.states.explorer import Exploration, ExplorationSettings, Explorer, Frontier
from datenwissenschaften.states.state import State
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

    def detect(self, frame):
        pass

    def distance(self, position):
        return 10.0 if self.seen else None


class _FakeMemory:
    def features(self, coordinates):
        return [0.0]

    def remember(self, coordinates):
        return True


class _ConcreteExplorer(Explorer[_FakeRam]):
    template_file = "unused.png"

    def _target_state(self):
        return State


def _explorer(monkeypatch):
    monkeypatch.setattr(target_module, "TemplateDetector", lambda template_file: _FakeDetector(template_file))
    monkeypatch.setattr(TargetMemory, "shared", classmethod(lambda cls, key, *, origin, scale: _FakeMemory()))
    explorer = _ConcreteExplorer()
    explorer.ram = _FakeRam()
    explorer.frame = None
    explorer.target_missing_steps = 0
    return explorer


def test_exploration_settings_rejects_a_non_positive_screen_size():
    with pytest.raises(ValueError, match="Screen size must be positive"):
        ExplorationSettings.create(0, 0.05)


def test_revisit_penalty_is_zero_when_the_staleness_limit_never_exceeds_the_grace_period():
    settings = ExplorationSettings.create(1, 0.05)
    frontier = Frontier.start((0, 0))

    assert frontier.revisit_penalty(100, settings) == 0.0


def test_completion_reward_is_at_least_the_configured_area_reward():
    settings = ExplorationSettings.create(100, 0.05)
    exploration = Exploration.start(Position(0, 0), settings)

    assert exploration.completion_reward() == settings.area_reward


def test_on_reset_builds_a_fresh_exploration_state(monkeypatch):
    explorer = _explorer(monkeypatch)

    explorer._on_reset()

    assert explorer.target_missing_steps == 0
    assert isinstance(explorer.exploration, Exploration)


def test_target_reward_adds_exploration_and_completion_bonus_once_target_is_found(monkeypatch):
    explorer = _explorer(monkeypatch)
    explorer._on_reset()
    explorer.target_detector.seen = True

    reward = explorer._target_reward(10.0)

    assert reward > 0.0


def test_target_reward_without_a_visible_target_skips_the_completion_bonus(monkeypatch):
    explorer = _explorer(monkeypatch)
    explorer._on_reset()
    explorer.target_detector.seen = False

    reward = explorer._target_reward(None)

    assert reward < 0.0


def test_auxiliary_features_pads_with_zero_frontier_features_before_first_reset(monkeypatch):
    explorer = _explorer(monkeypatch)

    features = explorer.auxiliary_features(explorer.ram)

    assert features[-5:] == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_auxiliary_features_reports_frontier_progress_after_reset(monkeypatch):
    explorer = _explorer(monkeypatch)
    explorer._on_reset()

    features = explorer.auxiliary_features(explorer.ram)

    assert len(features) == 6


def test_next_state_switches_to_the_target_state_once_seen(monkeypatch):
    explorer = _explorer(monkeypatch)
    explorer.target_detector.seen = False

    assert explorer._next() is None

    explorer.target_detector.seen = True

    assert explorer._next() is State


def test_won_is_always_false(monkeypatch):
    explorer = _explorer(monkeypatch)

    assert explorer._won() is False


def test_target_state_abstract_stub_has_no_body(monkeypatch):
    explorer = _explorer(monkeypatch)

    assert Explorer._target_state(explorer) is None
