import threading
from pathlib import Path
from typing import Any

import cv2
import msgspec
import numpy as np

from datenwissenschaften.knowledge import Knowledge, knowledge_field, sprite_names
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.journal import now
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.sprites.objects import image_of, objects_in

SPRITES_DIR = "sprites"
CATALOG_FILE = "catalog.json"
SHOWN_SPRITES = 48


class Sighting(msgspec.Struct):
    first_seen: str
    first_level: int | None
    first_state: str
    seen: int


def sprites_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / SPRITES_DIR


def png(image: np.ndarray) -> bytes:
    return cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1].tobytes()


class SpriteCatalog:
    def __init__(self, directory: Path, knowledge: Knowledge) -> None:
        self.directory = directory
        self.knowledge = knowledge
        self.names = sprite_names(knowledge)
        self.lock = threading.Lock()
        catalog = directory / CATALOG_FILE
        self.sightings = (
            msgspec.json.decode(catalog.read_bytes(), type=dict[str, Sighting]) if catalog.is_file() else {}
        )
        self.images = {signature: self.image_path(signature).read_bytes() for signature in self.sightings}

    def image_path(self, signature: str) -> Path:
        return self.directory / f"{signature}.png"

    def look(self, state: bytes, screen: np.ndarray, phase: str, level: int | None) -> None:
        found = objects_in(state, screen)
        with self.lock:
            for shown in found:
                if shown.signature in self.sightings:
                    self.sightings[shown.signature].seen += 1
                    continue
                self.sightings[shown.signature] = Sighting(now(), level, phase, 1)
                self.images[shown.signature] = png(image_of(shown, screen))

    def save(self) -> None:
        with self.lock:
            images = dict(self.images)
            catalog = msgspec.json.encode(self.sightings)
        for signature, image in images.items():
            if not self.image_path(signature).is_file():
                atomic_write(self.image_path(signature), image)
        atomic_write(self.directory / CATALOG_FILE, catalog)

    def sprite_field(self, signature: str, sighting: Sighting) -> dict[str, Any]:
        return {
            "signature": signature,
            "name": self.names[signature] if signature in self.names else None,
            "first_seen": sighting.first_seen,
            "first_level": sighting.first_level,
            "first_state": sighting.first_state,
            "seen": sighting.seen,
        }

    def fields(self) -> dict[str, Any]:
        with self.lock:
            ranked = sorted(self.sightings.items(), key=lambda item: item[1].seen, reverse=True)
            sprites = [self.sprite_field(signature, sighting) for signature, sighting in ranked[:SHOWN_SPRITES]]
        return {"sprites": sprites, "knowledge": knowledge_field(self.knowledge)}
