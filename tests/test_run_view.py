import io
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
from PIL import Image

from datenwissenschaften.speedrun import footage
from datenwissenschaften.speedrun.footage import render_run, run_cut
from datenwissenschaften.speedrun.journal import Split
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import FULL_RUN
from datenwissenschaften.speedrun.run_panel import GAME_WIDTH, HEIGHT, WIDTH, Moment, annotated, sprite_colors
from datenwissenschaften.speedrun.sprite_matches import Match, sprite_matches, templates_of

COURSE = SimpleNamespace(first="PowerOn", frame_rate=60.0)
SEGMENTS = [Segment("Play", [[0, 1]] * 2, b"a", False), Segment("Goal", [[1, 0]] * 3, b"b", True)]


def arrow() -> tuple[np.ndarray, bytes]:
    rgba = np.zeros((6, 8, 4), dtype=np.uint8)
    rgba[2:4, :, :3] = (250, 200, 40)
    rgba[:, 5, :3] = (40, 120, 250)
    rgba[2:4, :, 3] = 255
    rgba[:, 5, 3] = 255
    buffer = io.BytesIO()
    Image.fromarray(rgba).save(buffer, format="PNG")
    return rgba, buffer.getvalue()


def pasted(screen: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    opaque = rgba[:, :, 3] > 0
    area = screen[y : y + rgba.shape[0], x : x + rgba.shape[1]]
    area[opaque] = rgba[:, :, :3][opaque]


def test_a_sprite_is_found_once_where_it_stands_facing_either_way():
    rgba, png = arrow()
    screen = np.random.default_rng(3).integers(0, 40, (240, 256, 3)).astype(np.uint8)
    pasted(screen, rgba, 40, 30)
    pasted(screen, rgba[:, ::-1], 100, 60)

    found = sprite_matches(screen, templates_of({"arrow": png}))

    assert sorted((match.x, match.y) for match in found) == [(40, 30), (100, 60)]
    assert {(match.name, match.width, match.height) for match in found} == {("arrow", 8, 6)}


def test_the_run_view_scales_the_game_and_boxes_every_sprite_in_its_own_colour():
    screen = np.zeros((240, 256, 3), dtype=np.uint8)
    screen[10, 20] = (255, 255, 255)
    colors = sprite_colors(["arrow", "coin"])
    moment = Moment(59, 1.0, "Level1", [Split("Intro", 0.5, 0.5)], {"Lives": 3, "X": None})

    frame = annotated(screen, [Match("coin", 50, 40, 8, 8)], colors, "Comet Courier", moment)

    assert frame.shape == (HEIGHT, WIDTH, 3) and len(set(colors.values())) == 2
    assert (frame[20:22, 40:42] == 255).all()
    assert tuple(frame[80, 100]) == colors["coin"]
    assert frame[:, GAME_WIDTH:].any()


def test_the_full_run_is_cut_from_its_pieces_and_rendered_at_low_priority(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []
    run_cut_shown = run_cut(COURSE, SEGMENTS)
    monkeypatch.setattr(
        footage.subprocess, "run", lambda command, check, capture_output, text: commands.append(command)
    )

    video = render_run("speedrun.walk:Wrapper", tmp_path / "config.yaml", tmp_path, run_cut_shown.recordings)

    assert video == tmp_path / "run.mp4"
    assert msgspec.msgpack.decode((tmp_path / "run.msgpack").read_bytes(), type=list[Segment]) == SEGMENTS
    assert commands[0][:3] == ["nice", "-n", "19"] and "datenwissenschaften.speedrun.run_view" in commands[0]
    assert (run_cut_shown.field["part"], run_cut_shown.field["frames"]) == (FULL_RUN, 5)
    assert run_cut(COURSE, SEGMENTS[:1]).key != run_cut_shown.key
