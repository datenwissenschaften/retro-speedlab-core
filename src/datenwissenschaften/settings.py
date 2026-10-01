from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from box import Box

MAX_PORT = 65_535


@dataclass(frozen=True)
class RetroSpeedlabPaths:
    roms_path: Path
    savestates_dir: Path
    models_dir: Path
    record_dir: Path
    cache_dir: Path
    database_path: Path
    reports_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    game: str
    game_identity: str
    savestates: tuple[str, ...]
    rotation_minutes: int
    fingerprint: str | None


@dataclass(frozen=True)
class LayaSettings:
    checkpoint: str


@dataclass(frozen=True)
class UploadSettings:
    url: str
    api_key: str | None


@dataclass(frozen=True)
class UISettings:
    enabled: bool
    host: str
    port: int
    max_episodes: int
    release: str
    persona: str
    twitch: bool
    dialogs: bool
    dialog_model: str


@dataclass(frozen=True)
class RetroSpeedlabConfig:
    paths: RetroSpeedlabPaths
    training: TrainingSettings
    laya: LayaSettings
    upload: UploadSettings
    ui: UISettings
    log_level: str


def load_config(config_path: Path) -> RetroSpeedlabConfig:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Configuration file not found: {config_path}")
    document = Box(yaml.safe_load(config_path.read_text(encoding="utf-8")), box_dots=False)
    base_dir = config_path.parent
    training = document.training
    ui = document.ui
    return RetroSpeedlabConfig(
        paths=RetroSpeedlabPaths(
            roms_path=_path(document.paths.roms, base_dir),
            savestates_dir=_path(document.paths.savestates, base_dir),
            models_dir=_path(document.paths.models, base_dir),
            record_dir=_path(document.paths.recordings, base_dir),
            cache_dir=_path(document.paths.cache, base_dir),
            database_path=_path(document.paths.database, base_dir),
            reports_dir=_path(document.paths.reports, base_dir),
        ),
        training=TrainingSettings(
            game=_text(training.game, "training.game"),
            game_identity=_text(training.game_identity, "training.game_identity")
            if "game_identity" in training
            else _text(training.game, "training.game"),
            savestates=_texts(training.savestates, "training.savestates"),
            rotation_minutes=_positive_int(training.rotation_minutes, "training.rotation_minutes"),
            fingerprint=_optional_text(training.fingerprint, "training.fingerprint"),
        ),
        laya=LayaSettings(checkpoint=_text(document.laya.checkpoint, "laya.checkpoint")),
        upload=UploadSettings(
            url=_text(document.upload.url, "upload.url"),
            api_key=_optional_text(document.upload.api_key, "upload.api_key"),
        ),
        ui=UISettings(
            enabled=_boolean(ui.enable, "ui.enable"),
            host=_text(ui.host, "ui.host"),
            port=_port(ui.port),
            max_episodes=_positive_int(ui.max_episodes, "ui.max_episodes"),
            release=_text(ui.release, "ui.release"),
            persona=_text(ui.persona, "ui.persona"),
            twitch=_boolean(document.twitch.enabled, "twitch.enabled"),
            dialogs=_boolean(document.twitch.dialogs, "twitch.dialogs"),
            dialog_model=_text(document.twitch.dialog_model, "twitch.dialog_model"),
        ),
        log_level=_text(document.log_level, "log_level").upper(),
    )


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Configuration value '{name}' must be a non-empty string.")
    return value.strip()


def _texts(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Configuration value '{name}' must be a non-empty list.")
    return tuple(_text(item, name) for item in value)


def _optional_text(value: Any, name: str) -> str | None:
    return None if value is None else _text(value, name)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"Configuration value '{name}' must be a boolean.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise RuntimeError(f"Configuration value '{name}' must be a positive integer.")
    return value


def _port(value: Any) -> int:
    port = _positive_int(value, "ui.port")
    if port > MAX_PORT:
        raise RuntimeError(f"Configuration value 'ui.port' must be at most {MAX_PORT}.")
    return port


def _path(value: Any, base_dir: Path) -> Path:
    path = Path(_text(value, "paths")).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
