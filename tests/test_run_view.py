from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np

from datenwissenschaften.speedrun import footage
from datenwissenschaften.speedrun.footage import render_run, run_cut
from datenwissenschaften.speedrun.journal import Split
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import FULL_RUN
from datenwissenschaften.speedrun.run_panel import GAME_WIDTH, HEIGHT, WIDTH, Moment, annotated

COURSE = SimpleNamespace(first="PowerOn", frame_rate=60.0)
SEGMENTS = [Segment("Play", [[0, 1]] * 2, b"a", False), Segment("Goal", [[1, 0]] * 3, b"b", True)]


def test_the_run_view_scales_the_game_and_writes_the_panel_beside_it():
    screen = np.zeros((240, 256, 3), dtype=np.uint8)
    screen[10, 20] = (255, 255, 255)
    moment = Moment(59, 1.0, "Level1", [Split("Intro", 0.5, 0.5)], {"Lives": 3, "X": None})

    frame = annotated(screen, "Comet Courier", moment)

    assert frame.shape == (HEIGHT, WIDTH, 3)
    assert (frame[20:22, 40:42] == 255).all()
    assert (frame[80:96, 100:116] == 0).all()
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
