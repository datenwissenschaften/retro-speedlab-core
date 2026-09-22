from types import SimpleNamespace

from datenwissenschaften.callbacks import model_metadata_callback as model_metadata_callback_module
from datenwissenschaften.callbacks.model_metadata_callback import ModelMetadataCallback


def _patch_publish(monkeypatch):
    published = []
    monkeypatch.setattr(
        model_metadata_callback_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append((section, values, replace)),
    )
    monkeypatch.setattr(
        model_metadata_callback_module,
        "get_runtime",
        lambda: SimpleNamespace(get_model_metadata=lambda model: {"steps": model.num_timesteps}),
    )
    return published


def test_interval_seconds_is_clamped_to_a_minimum():
    assert ModelMetadataCallback(interval_seconds=0.0).interval_seconds == 0.1
    assert ModelMetadataCallback(interval_seconds=-5.0).interval_seconds == 0.1
    assert ModelMetadataCallback(interval_seconds=2.5).interval_seconds == 2.5


def test_on_training_start_publishes_immediately(monkeypatch):
    published = _patch_publish(monkeypatch)
    monkeypatch.setattr(model_metadata_callback_module.time, "monotonic", lambda: 100.0)
    callback = ModelMetadataCallback(interval_seconds=5.0)
    callback.model = SimpleNamespace(num_timesteps=1)

    callback._on_training_start()

    assert published == [("model", {"steps": 1}, True)]
    assert callback._next_publish_at == 105.0


def test_on_step_skips_publish_before_interval_elapses(monkeypatch):
    published = _patch_publish(monkeypatch)
    monkeypatch.setattr(model_metadata_callback_module.time, "monotonic", lambda: 100.0)
    callback = ModelMetadataCallback(interval_seconds=5.0)
    callback.model = SimpleNamespace(num_timesteps=1)
    callback._next_publish_at = 200.0

    result = callback._on_step()

    assert result is True
    assert published == []
    assert callback._next_publish_at == 200.0


def test_on_step_publishes_once_interval_has_elapsed(monkeypatch):
    published = _patch_publish(monkeypatch)
    monkeypatch.setattr(model_metadata_callback_module.time, "monotonic", lambda: 300.0)
    callback = ModelMetadataCallback(interval_seconds=5.0)
    callback.model = SimpleNamespace(num_timesteps=7)
    callback._next_publish_at = 50.0

    result = callback._on_step()

    assert result is True
    assert published == [("model", {"steps": 7}, True)]
    assert callback._next_publish_at == 305.0


def test_on_rollout_end_publishes(monkeypatch):
    published = _patch_publish(monkeypatch)
    monkeypatch.setattr(model_metadata_callback_module.time, "monotonic", lambda: 10.0)
    callback = ModelMetadataCallback(interval_seconds=1.0)
    callback.model = SimpleNamespace(num_timesteps=9)

    callback._on_rollout_end()

    assert published == [("model", {"steps": 9}, True)]
