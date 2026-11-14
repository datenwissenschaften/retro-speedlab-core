from pathlib import Path
from typing import Any

import msgspec
import yaml

KNOWLEDGE_FILE = "knowledge.yaml"


class Mechanic(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    title: str
    text: str


class Player(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    text: str
    sprite: str | None


class Enemy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    text: str
    sprite: str | None
    levels: list[str | int]


class Powerup(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    text: str
    sprite: str | None
    effect: str


class Level(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    goal: str
    route: str
    hazards: list[str]


class RamValue(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    address: str | int
    name: str
    meaning: str
    verified: bool


class Knowledge(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    mechanics: list[Mechanic]
    player: Player | None
    enemies: list[Enemy]
    powerups: list[Powerup]
    levels: list[Level]
    ram: list[RamValue]


NO_KNOWLEDGE = Knowledge([], None, [], [], [], [])


def knowledge_path(game_dir: Path) -> Path:
    return game_dir / KNOWLEDGE_FILE


def load_knowledge(path: Path) -> Knowledge:
    if not path.is_file():
        return NO_KNOWLEDGE
    try:
        return msgspec.convert(yaml.safe_load(path.read_text(encoding="utf-8")), type=Knowledge)
    except (yaml.YAMLError, msgspec.ValidationError) as error:
        raise RuntimeError(f"{path} is not valid game knowledge: {error}") from error


def knowledge_field(knowledge: Knowledge) -> dict[str, Any]:
    return msgspec.to_builtins(knowledge)
