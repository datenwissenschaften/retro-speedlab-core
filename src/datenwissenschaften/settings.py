from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from box import Box
from speedlab_engine.collection import cartridge_name, listed, load_collection


@dataclass(frozen=True)
class RetroSpeedlabPaths:
    collection: Path
    models_dir: Path
    cache_dir: Path
    lab_run_marker: Path
    game_dir: Path


@dataclass(frozen=True)
class TrainingSettings:
    game: str
    cartridge: str


@dataclass(frozen=True)
class UploadSettings:
    url: str
    api_key: str | None


@dataclass(frozen=True)
class UISettings:
    enabled: bool
    release: str
    twitch: bool


@dataclass(frozen=True)
class LabSettings:
    stall_minutes: int


@dataclass(frozen=True)
class GameSettings:
    id: str
    number: int
    title: str
    version: str
    collection: str
    category: str


@dataclass(frozen=True)
class SpeedrunSettings:
    scouts: int
    backtrackers: int
    beam_scouts: int
    vision_scouts: int
    polishers: int
    segment_seconds: float
    policy: bool
    policy_emulators: int
    game: GameSettings


@dataclass(frozen=True)
class RetroSpeedlabConfig:
    paths: RetroSpeedlabPaths
    training: TrainingSettings
    upload: UploadSettings
    ui: UISettings
    lab: LabSettings
    speedrun: SpeedrunSettings
    log_level: str


def load_config(config_path: Path) -> RetroSpeedlabConfig:
    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise RuntimeError(f"Configuration file not found: {config_path}")
    document: Any = Box(yaml.safe_load(config_path.read_text(encoding="utf-8")), box_dots=False)
    base_dir = config_path.parent
    collection = _path(document.paths.collection, base_dir)
    slug = _text(document.training.game, "training.game")
    ui = document.ui
    return RetroSpeedlabConfig(
        paths=RetroSpeedlabPaths(
            collection=collection,
            models_dir=_path(document.paths.models, base_dir),
            cache_dir=_path(document.paths.cache, base_dir),
            lab_run_marker=_path(document.paths.lab_run, base_dir),
            game_dir=base_dir,
        ),
        training=TrainingSettings(game=slug, cartridge=cartridge_name(slug)),
        upload=UploadSettings(
            url=_text(document.upload.url, "upload.url"),
            api_key=_optional_text(document.upload.api_key, "upload.api_key"),
        ),
        ui=UISettings(
            enabled=_boolean(ui.enable, "ui.enable"),
            release=_text(ui.release, "ui.release"),
            twitch=_boolean(document.twitch.enabled, "twitch.enabled"),
        ),
        lab=LabSettings(stall_minutes=_positive_int(document.agent.stall_minutes, "agent.stall_minutes")),
        speedrun=SpeedrunSettings(
            scouts=_positive_int(document.speedrun.scouts, "speedrun.scouts"),
            backtrackers=_count(document.speedrun.backtrackers, "speedrun.backtrackers"),
            beam_scouts=_count(document.speedrun.beam_scouts, "speedrun.beam_scouts"),
            vision_scouts=_count(document.speedrun.vision_scouts, "speedrun.vision_scouts"),
            polishers=_positive_int(document.speedrun.polishers, "speedrun.polishers"),
            segment_seconds=_positive_number(document.speedrun.segment_seconds, "speedrun.segment_seconds"),
            policy=_boolean(document.speedrun.policy, "speedrun.policy"),
            policy_emulators=_positive_int(document.speedrun.policy_emulators, "speedrun.policy_emulators"),
            game=_game(collection, slug, document.speedrun),
        ),
        log_level=_text(document.log_level, "log_level").upper(),
    )


def _game(collection: Path, slug: str, speedrun: Any) -> GameSettings:
    games = load_collection(collection)
    game = listed(games, slug)
    category = _text(speedrun.category, "speedrun.category")
    return GameSettings(game.slug, game.number, game.title, game.version, games.title, category)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Configuration value '{name}' must be a non-empty string.")
    return value.strip()


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


def _count(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"Configuration value '{name}' must be zero or a positive integer.")
    return value


def _positive_number(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
        raise RuntimeError(f"Configuration value '{name}' must be a positive number.")
    return float(value)


def _path(value: Any, base_dir: Path) -> Path:
    path = Path(_text(value, "paths")).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()
