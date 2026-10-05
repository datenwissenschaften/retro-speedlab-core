from pathlib import Path
from typing import Any

import pytest
import yaml
from box.exceptions import BoxKeyError

from datenwissenschaften.settings import load_config


def _document() -> dict[str, Any]:
    return {
        "paths": {
            "roms": "roms",
            "integrations": "integrations",
            "models": "models",
            "recordings": "recordings",
            "cache": "cache",
            "database": "database.json",
            "reports": "agent/reports",
            "curriculum": "curriculum",
            "demonstrations": "demonstrations",
            "lab_run": "lab-run",
        },
        "training": {
            "game": "TestGame",
            "fingerprint": None,
        },
        "laya": {"checkpoint": "convaiinnovations/laya"},
        "upload": {"url": "https://example.test", "api_key": None},
        "ui": {
            "enable": True,
            "host": "127.0.0.1",
            "port": 18080,
            "max_episodes": 1000,
            "release": "local",
            "persona": "Retra",
        },
        "twitch": {"enabled": True, "summary_models": ["test/model:free"]},
        "log_level": "info",
    }


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def test_complete_config_loads(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert config.training.game == "TestGame"
    assert config.training.game_identity == "TestGame"
    assert config.training.fingerprint is None
    assert config.laya.checkpoint == "convaiinnovations/laya"
    assert config.ui.port == 18080
    assert config.log_level == "INFO"


def test_paths_resolve_relative_to_the_config_file(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert config.paths.roms_path == (tmp_path / "roms").resolve()
    assert config.paths.integrations_dir == (tmp_path / "integrations").resolve()
    assert config.paths.curriculum_dir == (tmp_path / "curriculum").resolve()
    assert config.paths.demonstrations_dir == (tmp_path / "demonstrations").resolve()
    assert config.paths.record_dir == (tmp_path / "recordings").resolve()
    assert config.paths.database_path == (tmp_path / "database.json").resolve()


def test_explicit_game_identity_wins(tmp_path: Path):
    document = _document()
    document["training"]["game_identity"] = "TestGame-v2"

    assert load_config(_write(tmp_path, document)).training.game_identity == "TestGame-v2"


def test_missing_laya_checkpoint_fails_fast(tmp_path: Path):
    document = _document()
    del document["laya"]

    with pytest.raises(BoxKeyError):
        load_config(_write(tmp_path, document))


@pytest.mark.parametrize("port", [0, 70_000, True, "18080"])
def test_invalid_ui_port_is_rejected(tmp_path: Path, port: Any):
    document = _document()
    document["ui"]["port"] = port

    with pytest.raises(RuntimeError, match="ui.port"):
        load_config(_write(tmp_path, document))


def test_missing_config_file_is_reported(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_twitch_must_be_switched_on_or_off_explicitly(tmp_path: Path):
    document = _document()
    document["twitch"]["enabled"] = "yes"

    with pytest.raises(RuntimeError, match="twitch.enabled"):
        load_config(_write(tmp_path, document))
