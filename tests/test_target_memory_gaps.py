from types import SimpleNamespace

import pytest

from datenwissenschaften.states import target_memory as target_memory_module
from datenwissenschaften.states.target_memory import TargetMemory


class _FakeRedisStore:
    def __init__(self, redis_url, *, key_prefix="datenwissenschaften"):
        self.values: dict[tuple, object] = {}

    def get(self, *parts, default=None):
        return self.values.get(parts, default)

    def set(self, *parts, value):
        self.values[parts] = value

    def delete(self, *parts):
        self.values.pop(parts, None)


def _configure(monkeypatch, *, active_savestate="Level1"):
    monkeypatch.setattr(TargetMemory, "_registry", {})
    store = _FakeRedisStore("redis://example")
    monkeypatch.setattr(target_memory_module, "RedisStore", lambda url, **kwargs: store)
    monkeypatch.setattr(
        target_memory_module,
        "load_config",
        lambda: SimpleNamespace(
            ui=SimpleNamespace(redis_url="redis://example"),
            training=SimpleNamespace(game_identity="Game", active_savestate=active_savestate),
        ),
    )
    return store


def test_shared_rejects_an_incompatible_schema_for_an_existing_key(monkeypatch):
    _configure(monkeypatch)
    TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)

    with pytest.raises(ValueError, match="incompatible coordinate schema"):
        TargetMemory.shared("Target", origin=(0.0, 0.0, 0.0), scale=100.0)


def test_remember_is_a_no_op_once_a_coordinate_is_already_known(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.remember((10.0, 20.0))

    assert memory.remember((30.0, 40.0)) is False
    assert memory.coordinates == (10.0, 20.0)


def test_features_with_no_remembered_coordinates_reports_unknown(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)

    features = memory.features((5.0, 5.0))

    assert features[0] == 0.0
    assert len(features) == 3


def test_features_defaults_current_coordinates_to_the_origin(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.remember((10.0, 20.0))

    features = memory.features()

    assert features[0] == 1.0
    assert len(features) == 3


def test_features_reports_a_known_target_with_scaled_deltas(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.remember((100.0, 0.0))

    features = memory.features((0.0, 0.0))

    assert features[0] == 1.0
    assert features[1] > 0.0


def test_validate_dimensions_rejects_the_wrong_coordinate_count(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)

    with pytest.raises(ValueError, match="Expected 2 target coordinates"):
        memory._validate_dimensions((1.0, 2.0, 3.0))


def test_save_is_a_no_op_when_there_are_no_coordinates_to_persist(monkeypatch):
    _configure(monkeypatch)
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.coordinates = None

    memory._save()

    assert memory._store.get(*memory._scope) is None


def test_coordinates_rejects_an_empty_sequence():
    with pytest.raises(ValueError, match="at least one coordinate dimension"):
        TargetMemory._coordinates(())


def test_scale_accepts_a_per_dimension_sequence():
    assert TargetMemory._scale([1.0, 2.0], 2) == (1.0, 2.0)


def test_scale_rejects_a_mismatched_or_non_positive_sequence():
    with pytest.raises(ValueError, match="one positive value per coordinate dimension"):
        TargetMemory._scale([1.0], 2)

    with pytest.raises(ValueError, match="one positive value per coordinate dimension"):
        TargetMemory._scale([1.0, -2.0], 2)
