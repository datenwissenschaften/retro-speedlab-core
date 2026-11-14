from pathlib import Path
from typing import Any

import pytest
import yaml
from box.exceptions import BoxKeyError
from speedlab_engine.collection import DEMO_COLLECTION

from datenwissenschaften.settings import GameSettings, SpeedrunSettings, load_config


def _document() -> dict[str, Any]:
    return {
        "paths": {
            "collection": str(DEMO_COLLECTION),
            "models": "models",
            "cache": "cache",
            "lab_run": "lab-run",
        },
        "training": {"game": "walk"},
        "upload": {"url": "https://example.test", "api_key": None},
        "ui": {"enable": True, "live": True, "release": "local"},
        "twitch": {"enabled": True},
        "agent": {"stall_minutes": 30},
        "speedrun": {
            "scouts": 3,
            "backtrackers": 2,
            "beam_scouts": 2,
            "vision_scouts": 1,
            "polishers": 2,
            "segment_seconds": 60,
            "policy": False,
            "policy_emulators": 3,
            "category": "Warpless",
        },
        "log_level": "info",
    }


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def test_complete_config_loads(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert (config.training.game, config.training.cartridge) == ("walk", "games.walk.cartridge")
    assert config.lab.stall_minutes == 30
    game = GameSettings("walk", 1, "Walk", "1.0.0", "DEMO", "Warpless")
    assert config.speedrun == SpeedrunSettings(3, 2, 2, 1, 2, 60.0, False, 3, game)
    assert config.upload.api_key is None
    assert config.log_level == "INFO"


def test_paths_resolve_relative_to_the_config_file(tmp_path: Path):
    config = load_config(_write(tmp_path, _document()))

    assert config.paths.collection == DEMO_COLLECTION.resolve()
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


def test_keys_the_core_no_longer_reads_are_ignored(tmp_path: Path):
    document = _document()
    document["speedrun"] |= {"guide": True, "speedrun_com": {"game": "m1z0mmd0"}}

    assert load_config(_write(tmp_path, document)) == load_config(_write(tmp_path, _document()))


def test_the_game_must_be_listed_in_the_collection(tmp_path: Path):
    document = _document()
    document["training"]["game"] = "comet_courier"

    with pytest.raises(LookupError, match="DEMO lists 0 games named comet_courier"):
        load_config(_write(tmp_path, document))


@pytest.mark.parametrize(
    ("section", "key", "value", "named"),
    [
        ("speedrun", "policy", "no", "speedrun.policy"),
        ("speedrun", "scouts", 0, "speedrun.scouts"),
        ("speedrun", "backtrackers", -1, "speedrun.backtrackers"),
        ("speedrun", "segment_seconds", 0, "speedrun.segment_seconds"),
        ("upload", "api_key", " ", "upload.api_key"),
    ],
)
def test_a_malformed_value_is_named_when_the_config_loads(tmp_path: Path, section: str, key: str, value, named: str):
    document = _document()
    document[section][key] = value

    with pytest.raises(RuntimeError, match=named):
        load_config(_write(tmp_path, document))
