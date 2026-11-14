import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speedlab_engine.sprites import sprite_images

from datenwissenschaften.knowledge import Knowledge, knowledge_field, knowledge_path, load_knowledge
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.paths import atomic_write

SPRITES_DIR = "sprites"
PNG = ".png"


def sprites_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / SPRITES_DIR / settings.speedrun.game.version


def rendered_sprites(settings: RetroSpeedlabConfig) -> dict[str, bytes]:
    directory = sprites_dir(settings)
    if not directory.is_dir():
        rendering = directory.with_name(f".{directory.name}")
        rendering.mkdir(parents=True, exist_ok=True)
        for name, png in sprite_images(settings.paths.collection, settings.training.cartridge).items():
            atomic_write(rendering / f"{name}{PNG}", png)
        rendering.rename(directory)
    return {path.stem: path.read_bytes() for path in sorted(directory.glob(f"*{PNG}"))}


def sprite_field(images: dict[str, bytes]) -> list[dict[str, str]]:
    return [
        {"signature": name, "name": name, "png": base64.b64encode(png).decode("ascii")} for name, png in images.items()
    ]


@dataclass(slots=True, frozen=True)
class Learned:
    knowledge: Knowledge
    sprites: list[dict[str, str]]

    def fields(self) -> dict[str, Any]:
        return {"knowledge": knowledge_field(self.knowledge), "sprites": self.sprites}


def learned_of(settings: RetroSpeedlabConfig) -> Learned:
    return Learned(load_knowledge(knowledge_path(settings.paths.game_dir)), sprite_field(rendered_sprites(settings)))
