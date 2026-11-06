from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import HIDDEN_SPRITE, NO_SPRITES, fceu_state

from datenwissenschaften.knowledge import NO_KNOWLEDGE, Enemy, Knowledge, Player, knowledge_path, load_knowledge
from datenwissenschaften.sprites.catalog import CATALOG_FILE, DATA_URI, SpriteCatalog
from datenwissenschaften.sprites.oam import TILE, sprites_of
from datenwissenschaften.sprites.objects import objects_in
from datenwissenschaften.sprites.overlay import (
    NAMED_COLOR,
    PIXEL_SCALE,
    PLAYER_COLOR,
    UNKNOWN_COLOR,
    annotate,
    labels_of,
)

SCREEN = (224, 240, 3)
KNOWLEDGE = """
mechanics: [{title: Weight, text: Eat nibbleys to get heavy enough for the scale.}]
player: {name: Rattle, text: The red snake., sprite: SNAKE}
enemies: [{name: Bigfoot, text: Stomps the snake., sprite: null, levels: [1, Level 2]}]
powerups: [{name: Clock, text: Freezes time., sprite: null, effect: More time}]
levels: [{name: Level 1, goal: Ring the bell, route: Up the stairs, hazards: [water]}]
ram: [{address: 0x00A5, name: weight, meaning: Pieces eaten, verified: true}]
"""


def oam(*sprites: tuple[int, int, int, int]) -> bytes:
    shown = b"".join(bytes(sprite) for sprite in sprites)
    return shown + HIDDEN_SPRITE * (64 - len(sprites))


def two_objects() -> bytes:
    snake = [(15, 1, 0, 40), (15, 2, 1, 48), (23, 3, 0, 40)]
    return fceu_state(b"cpu", oam(*snake, (99, 9, 2, 200)), 0)


def screen() -> np.ndarray:
    frame = np.zeros(SCREEN, np.uint8)
    frame[8:24, 32:48] = (200, 30, 30)
    return frame


def test_the_sprites_come_from_the_ppu_object_memory_in_the_emulator_state():
    sprites = sprites_of(fceu_state(b"cpu", oam((15, 7, 0x43, 40)), 0x20))

    assert [(sprite.x, sprite.y, sprite.height, sprite.tile, sprite.palette) for sprite in sprites] == [
        (40, 16, 2 * TILE, 7, 3)
    ]
    assert sprites_of(fceu_state(b"cpu", NO_SPRITES, 0)) == []
    with pytest.raises(ValueError, match="SPRA"):
        sprites_of(b"emulator")


def test_touching_sprites_form_one_object_placed_on_the_cropped_screen():
    objects = objects_in(two_objects(), screen())

    assert [(shown.left, shown.top, shown.right, shown.bottom) for shown in objects] == [
        (32, 8, 48, 24),
        (192, 92, 200, 100),
    ]
    assert objects[0].signature != objects[1].signature
    moved = fceu_state(b"cpu", oam((25, 3, 0, 80), (25, 2, 1, 88), (17, 1, 0, 80)), 0)
    assert objects_in(moved, screen())[0].signature == objects[0].signature
    offscreen = fceu_state(b"cpu", oam((15, 1, 0, 0)), 0)
    assert objects_in(offscreen, screen()) == []


def test_the_catalog_keeps_every_object_once_with_its_first_sighting_and_image(tmp_path: Path):
    knowledge = Knowledge([], Player("Rattle", "The snake.", "x"), [], [], [], [])
    catalog = SpriteCatalog(tmp_path / "sprites", knowledge)

    catalog.look(two_objects(), screen(), "Play", 1)
    catalog.look(fceu_state(b"cpu", oam((15, 1, 0, 40), (15, 2, 1, 48), (23, 3, 0, 40)), 0), screen(), "Door", 2)
    catalog.save()
    catalog.save()
    reloaded = SpriteCatalog(tmp_path / "sprites", NO_KNOWLEDGE).fields()

    snake, other = reloaded["sprites"]
    assert (snake["seen"], snake["first_level"], snake["first_state"], snake["name"]) == (2, 1, "Play", None)
    assert other["seen"] == 1 and snake["image"].startswith(DATA_URI)
    image = cv2.imread(str(tmp_path / "sprites" / f"{snake['signature']}.png"))
    assert image.shape == (16, 16, 3) and (tmp_path / "sprites" / CATALOG_FILE).is_file()
    assert reloaded["knowledge"]["player"] is None


def test_the_catalog_names_the_sprites_the_game_knowledge_names(tmp_path: Path):
    signature = objects_in(two_objects(), screen())[0].signature
    catalog = SpriteCatalog(tmp_path, Knowledge([], Player("Rattle", "The snake.", signature), [], [], [], []))

    catalog.look(two_objects(), screen(), "Play", None)

    fields = catalog.fields()
    assert [sprite["name"] for sprite in fields["sprites"]] == ["Rattle", None]
    assert fields["knowledge"]["player"] == {"name": "Rattle", "text": "The snake.", "sprite": signature}


def test_the_overlay_boxes_the_player_and_marks_unknown_objects(tmp_path: Path):
    snake, other = objects_in(two_objects(), screen())
    player = Player("Rattle", "The snake.", snake.signature)
    enemy = Enemy("Bigfoot", "Stomps.", other.signature, [])

    unknown = annotate(screen(), [snake, other], labels_of(Knowledge([], player, [], [], [], [])))
    named = annotate(screen(), [snake, other], labels_of(Knowledge([], player, [enemy], [], [], [])))

    assert unknown.shape == (SCREEN[0] * PIXEL_SCALE, SCREEN[1] * PIXEL_SCALE, 3)
    assert tuple(unknown[snake.bottom * PIXEL_SCALE - 1, 40 * PIXEL_SCALE]) == PLAYER_COLOR
    assert tuple(unknown[other.bottom * PIXEL_SCALE - 1, 196 * PIXEL_SCALE]) == UNKNOWN_COLOR
    assert tuple(named[other.bottom * PIXEL_SCALE - 1, 196 * PIXEL_SCALE]) == NAMED_COLOR
    assert tuple(unknown[20 * PIXEL_SCALE, 40 * PIXEL_SCALE]) == (200, 30, 30)


def test_the_game_knowledge_is_read_next_to_the_config(tmp_path: Path):
    knowledge_path(tmp_path).write_text(KNOWLEDGE, encoding="utf-8")

    knowledge = load_knowledge(knowledge_path(tmp_path))

    assert knowledge.player == Player("Rattle", "The red snake.", "SNAKE")
    assert knowledge.enemies[0].levels == [1, "Level 2"] and knowledge.ram[0].address == 0xA5
    assert labels_of(knowledge).names == {"SNAKE": "Rattle"}
    assert load_knowledge(tmp_path / "missing.yaml") == NO_KNOWLEDGE


@pytest.mark.parametrize("text", ["mechanics: [", "mechanics: []\nplayer: null", "- just a list", ""])
def test_malformed_game_knowledge_fails_with_the_file_named(tmp_path: Path, text: str):
    knowledge_path(tmp_path).write_text(text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="knowledge.yaml is not valid game knowledge"):
        load_knowledge(knowledge_path(tmp_path))
