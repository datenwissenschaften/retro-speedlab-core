import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fakes import write_config
from PIL import Image

from datenwissenschaften.knowledge import Player, load_knowledge
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun import learned
from datenwissenschaften.speedrun.learned import learned_of, rendered_sprites, sprites_dir


def test_the_game_sprites_render_once_per_version_and_are_published_as_named_pngs(tmp_path: Path, monkeypatch):
    settings = load_config(write_config(tmp_path))

    first = rendered_sprites(settings)
    monkeypatch.setattr(learned, "sprite_images", lambda collection, name: {"never": b""})
    again = learned_of(settings)

    assert first == rendered_sprites(settings)
    assert sprites_dir(settings) == tmp_path / "cache" / "sprites" / "1.0.0"
    assert [(sprite["signature"], sprite["name"]) for sprite in again.sprites] == [("dot", "dot")]
    dot = np.asarray(Image.open(io.BytesIO(base64.b64decode(again.sprites[0]["png"]))))
    assert dot.shape == (4, 4, 4) and dot[0, 0, 3] == 0 and dot[1, 1, 3] == 255
    assert again.fields()["sprites"] == again.sprites and again.fields()["knowledge"]["player"] is None


def test_knowledge_names_a_sprite_for_the_player_and_a_malformed_file_stops_the_run(tmp_path: Path):
    path = tmp_path / "knowledge.yaml"
    player = "{name: Dot, text: The dot., sprite: dot}"
    path.write_text(f"{{mechanics: [], player: {player}, enemies: [], powerups: [], levels: [], ram: []}}")
    assert load_knowledge(path).player == Player("Dot", "The dot.", "dot")
    path.write_text("{mechanics: []}")
    with pytest.raises(RuntimeError, match="not valid game knowledge"):
        load_knowledge(path)
