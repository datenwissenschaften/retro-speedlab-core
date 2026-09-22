from pathlib import Path
from typing import Any

import pytest
import yaml

from datenwissenschaften.settings import empty_all_paths, load_config, load_paths_from_config


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


def _write(tmp_path: Path, document: dict[str, Any]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return config_path


def _write_config(tmp_path: Path, **kwargs: Any) -> Path:
    return _write(tmp_path, _document(**kwargs))


def test_non_mapping_document_is_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(RuntimeError, match="YAML mapping"):
        load_config(config_path)


def test_non_mapping_section_is_rejected(tmp_path: Path):
    document = _document()
    document["training"] = ["not", "a", "mapping"]

    with pytest.raises(RuntimeError, match="'training' must be a mapping"):
        load_config(_write(tmp_path, document))


def test_missing_required_string_is_rejected(tmp_path: Path):
    document = _document()
    del document["training"]["game"]

    with pytest.raises(RuntimeError, match="Missing required configuration value: game"):
        load_config(_write(tmp_path, document))


def test_blank_string_value_is_rejected(tmp_path: Path):
    document = _document(training={"game": "   "})

    with pytest.raises(RuntimeError, match="'game' must be a non-empty string"):
        load_config(_write(tmp_path, document))


def test_missing_nullable_key_is_rejected(tmp_path: Path):
    document = _document()
    del document["upload"]["api_key"]

    with pytest.raises(RuntimeError, match="Missing required configuration value: api_key"):
        load_config(_write(tmp_path, document))


def test_null_savestates_list_is_treated_as_empty(tmp_path: Path):
    document = _document()
    del document["training"]["savestate"]
    document["training"]["savestates"] = None

    with pytest.raises(RuntimeError, match="training.savestate or training.savestates"):
        load_config(_write(tmp_path, document))


def test_non_list_savestates_is_rejected(tmp_path: Path):
    document = _document(training={"savestates": "Level1"})

    with pytest.raises(RuntimeError, match="'savestates' must be a list"):
        load_config(_write(tmp_path, document))


def test_savestates_list_with_blank_entry_is_rejected(tmp_path: Path):
    document = _document(training={"savestates": ["Level1", "  "]})

    with pytest.raises(RuntimeError, match="'savestates' must be a list"):
        load_config(_write(tmp_path, document))


def test_invalid_ui_string_shorthand_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui="sideways")

    with pytest.raises(RuntimeError, match="'enable' or 'disable'"):
        load_config(config_path)


def test_non_mapping_ui_value_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui=["enable"])

    with pytest.raises(RuntimeError, match="string, boolean, or mapping"):
        load_config(config_path)


def test_non_boolean_ui_enable_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui={"enable": "yes"})

    with pytest.raises(RuntimeError, match="'ui.enable' must be a boolean"):
        load_config(config_path)


def test_blank_ui_host_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui={"enable": True, "host": "   "})

    with pytest.raises(RuntimeError, match="'ui.host' must be a non-empty string"):
        load_config(config_path)


def test_blank_ui_redis_url_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui={"enable": True, "redis_url": ""})

    with pytest.raises(RuntimeError, match="'ui.redis_url' must be a non-empty string"):
        load_config(config_path)


def test_blank_ui_history_key_prefix_is_rejected(tmp_path: Path):
    config_path = _write_config(tmp_path, ui={"enable": True, "history_key_prefix": " "})

    with pytest.raises(RuntimeError, match="'ui.history_key_prefix' must be a non-empty string"):
        load_config(config_path)


def test_load_paths_from_config_returns_only_the_paths_section(tmp_path: Path):
    paths = load_paths_from_config(_write_config(tmp_path))

    assert paths.models_dir == (tmp_path / "models").resolve()


def test_empty_all_paths_removes_directories_and_stray_files(tmp_path: Path):
    config_path = _write_config(tmp_path)
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "artifact").write_text("stale", encoding="utf-8")
    record_dir = tmp_path / "recordings"
    record_dir.write_text("not actually a directory", encoding="utf-8")
    (tmp_path / "cache").mkdir()

    empty_all_paths(config_path)

    assert not models_dir.exists()
    assert not record_dir.exists()
    assert not (tmp_path / "cache").exists()
