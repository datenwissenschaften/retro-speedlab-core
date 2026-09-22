from types import SimpleNamespace

import pytest

from datenwissenschaften import runtime as runtime_module
from datenwissenschaften.runtime import RetroSpeedlabRuntime, configure_runtime, get_runtime


def _runtime(paths):
    return RetroSpeedlabRuntime(
        paths=paths,
        wrappers={},
        ignored_states={},
        default_states={},
        obs_size=(96, 96),
        get_game=lambda: "Game",
        get_savestate=lambda: "Level1",
        set_savestate=lambda savestate: None,
        get_state_value=lambda name: None,
        set_state_value=lambda name, value: None,
        get_model_path=lambda selected_game: "path",
        get_model_metadata=lambda model: {},
    )


def test_get_runtime_before_configuration_raises(monkeypatch):
    monkeypatch.setattr(runtime_module, "_runtime", None)

    with pytest.raises(RuntimeError, match="not configured"):
        get_runtime()


def test_configure_and_get_runtime_round_trip(monkeypatch, tmp_path):
    paths = SimpleNamespace(
        models_dir=tmp_path / "models",
        record_dir=tmp_path / "record",
        cache_dir=tmp_path / "cache",
    )
    runtime = _runtime(paths)

    configure_runtime(runtime)

    configured = get_runtime()
    assert configured is runtime
    assert configured.game == "Game"
    assert configured.savestate == "Level1"
    assert configured.models_dir == paths.models_dir
    assert configured.record_dir == paths.record_dir
    assert configured.cache_dir == paths.cache_dir
