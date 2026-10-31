from pathlib import Path
from typing import Any

import pytest
import yaml
from box.exceptions import BoxKeyError

from datenwissenschaften.settings import GameSettings, SpeedrunComSource, SpeedrunSettings, load_config


def _document() -> dict[str, Any]:
    return {
        "paths": {
            "roms": "roms",
            "integrations": "integrations",
            "models": "models",
            "cache": "cache",
            "lab_run": "lab-run",
        },
        "training": {"game": "TestGame"},
        "upload": {"url": "https://example.test", "api_key": None},
        "ui": {"enable": True, "release": "local"},
        "twitch": {"enabled": True},
        "agent": {"stall_minutes": 30},
        "speedrun": {
            "workers": 4,
            "segment_seconds": 60,
            "guide": False,
            "id": "testgame-v0",
            "title": "Test Game",
            "platform": "NES",
            "category": "Warpless",
            "speedrun_com": {"game": "m1z0mmd0", "category": "5dwpnllk", "variables": {"p85pg00l": "qyz5gr71"}},
            "tasvideos": 1694,
        },
        "log_level": "info",
    }


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def test_complete_config_loads(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert config.training.game == "TestGame"
    assert config.lab.stall_minutes == 30
    source = SpeedrunComSource("m1z0mmd0", "5dwpnllk", {"p85pg00l": "qyz5gr71"})
    game = GameSettings("testgame-v0", "Test Game", "NES", "Warpless", source, 1694)
    assert config.speedrun == SpeedrunSettings(4, 60.0, False, game)
    assert config.upload.api_key is None
    assert config.log_level == "INFO"


def test_paths_resolve_relative_to_the_config_file(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert config.paths.roms_path == (tmp_path / "roms").resolve()
    assert config.paths.integrations_dir == (tmp_path / "integrations").resolve()
    assert config.paths.models_dir == (tmp_path / "models").resolve()
    assert config.paths.cache_dir == (tmp_path / "cache").resolve()
    assert config.paths.lab_run_marker == (tmp_path / "lab-run").resolve()


def test_missing_speedrun_settings_fail_fast(tmp_path: Path):
    document = _document()
    del document["speedrun"]

    with pytest.raises(BoxKeyError):
        load_config(_write(tmp_path, document))


def test_missing_config_file_is_reported(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not found"):
        load_config(tmp_path / "missing.yaml")


def test_twitch_must_be_switched_on_or_off_explicitly(tmp_path: Path):
    document = _document()
    document["twitch"]["enabled"] = "yes"

    with pytest.raises(RuntimeError, match="twitch.enabled"):
        load_config(_write(tmp_path, document))


def test_the_record_sources_are_named_explicitly(tmp_path: Path):
    document = _document()
    document["speedrun"]["tasvideos"] = "1694"

    with pytest.raises(RuntimeError, match="speedrun.tasvideos"):
        load_config(_write(tmp_path, document))
