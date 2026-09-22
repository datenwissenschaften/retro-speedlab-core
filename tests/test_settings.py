from pathlib import Path
from typing import Any

import pytest
import yaml

from datenwissenschaften.settings import load_config


def _document(training: dict[str, Any] | None = None, ui: Any = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "paths": {
            "roms": "roms",
            "models": "models",
            "recordings": "recordings",
            "cache": "cache",
        },
        "training": {"game": "TestGame", "savestate": "Level1", "num_envs": 1, **(training or {})},
        "log_level": "INFO",
        "upload": {"url": "https://example.test", "api_key": None},
    }
    if ui is not None:
        document["ui"] = ui
    return document


def _write_config(tmp_path: Path, **kwargs: Any) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_document(**kwargs)), encoding="utf-8")
    return config_path


def test_minimal_config_loads_with_ui_defaults(tmp_path: Path):
    config = load_config(_write_config(tmp_path))

    assert config.training.game == "TestGame"
    assert config.training.game_identity == "TestGame"
    assert config.training.active_savestate == "Level1"
    assert config.ui.enabled is False
    assert config.ui.host == "127.0.0.1"
    assert config.ui.port == 18_080


def test_paths_resolve_relative_to_the_config_file(tmp_path: Path):
    config = load_config(_write_config(tmp_path))

    assert config.paths.models_dir == (tmp_path / "models").resolve()
    assert config.paths.roms_path == (tmp_path / "roms").resolve()


def test_missing_savestate_and_savestates_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    document = _document()
    del document["training"]["savestate"]
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(RuntimeError, match="training.savestate or training.savestates"):
        load_config(config_path)


def test_savestates_list_takes_priority_over_active_savestate(tmp_path: Path):
    config = load_config(_write_config(tmp_path, training={"savestates": ["Level2", "Level3"]}))

    assert config.training.active_savestate == "Level2"
    assert config.training.savestates == ("Level2", "Level3")


def test_duplicate_savestates_are_deduplicated(tmp_path: Path):
    config = load_config(_write_config(tmp_path, training={"savestates": ["Level2", "Level2", "Level3"]}))

    assert config.training.savestates == ("Level2", "Level3")


def test_missing_config_file_raises_a_clear_error(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Configuration file not found"):
        load_config(tmp_path / "missing.yaml")


def test_invalid_yaml_raises_a_clear_error(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training: [unterminated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Invalid YAML"):
        load_config(config_path)


def test_num_envs_accepts_auto_and_delegates_to_parallelism(tmp_path: Path, monkeypatch):
    import datenwissenschaften.settings as settings

    monkeypatch.setattr(settings, "optimal_env_count", lambda: 3)
    config = load_config(_write_config(tmp_path, training={"num_envs": "auto"}))

    assert config.training.num_envs == 3


@pytest.mark.parametrize("num_envs", [0, -1, 2.5, True])
def test_invalid_num_envs_is_rejected(tmp_path: Path, num_envs):
    config_path = _write_config(tmp_path, training={"num_envs": num_envs})

    with pytest.raises(RuntimeError, match="num_envs"):
        load_config(config_path)


def test_ui_string_shorthand_toggles_enabled(tmp_path: Path):
    enabled = load_config(_write_config(tmp_path, ui="enable"))
    disabled = load_config(_write_config(tmp_path, ui="disable"))

    assert enabled.ui.enabled is True
    assert disabled.ui.enabled is False


def test_ui_boolean_shorthand_toggles_enabled(tmp_path: Path):
    config = load_config(_write_config(tmp_path, ui=True))

    assert config.ui.enabled is True


def test_ui_enable_and_legacy_enabled_together_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui={"enable": True, "enabled": True})

    with pytest.raises(RuntimeError, match="ui.enable"):
        load_config(config_path)


@pytest.mark.parametrize("port", [0, -1, 65_536, 1.5, True])
def test_invalid_ui_port_is_rejected(tmp_path: Path, port):
    config_path = _write_config(tmp_path, ui={"enable": True, "port": port})

    with pytest.raises(RuntimeError, match="ui.port"):
        load_config(config_path)


@pytest.mark.parametrize("max_episodes", [0, -1, 1.5, True])
def test_invalid_max_episodes_is_rejected(tmp_path: Path, max_episodes):
    config_path = _write_config(tmp_path, ui={"enable": True, "max_episodes": max_episodes})

    with pytest.raises(RuntimeError, match="ui.max_episodes"):
        load_config(config_path)


def test_null_max_episodes_is_accepted(tmp_path: Path):
    config = load_config(_write_config(tmp_path, ui={"enable": True, "max_episodes": None}))

    assert config.ui.max_episodes is None


def test_shipped_example_config_loads_once_a_savestate_is_set(tmp_path: Path):
    # Regression test: config.example.yaml previously shipped with a
    # `paths.savestates` key that `load_config` never reads, and was missing
    # the required `paths.cache` key entirely, so `cp config.example.yaml
    # config.yaml` failed before training could even start.
    example_path = Path(__file__).resolve().parent.parent / "config.example.yaml"
    document = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    document["training"]["savestate"] = "Level1"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_config(config_path)

    assert config.training.game == "Airstriker-Genesis-v0"
    assert config.paths.cache_dir == (tmp_path / "working" / "cache").resolve()
