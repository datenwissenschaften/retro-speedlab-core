from pathlib import Path

import numpy as np
from fakes import write_config
from speedlab_engine.cartridge import BUTTONS
from speedlab_engine.collection import DEMO_COLLECTION, cartridge_name
from speedlab_engine.console import Console
from speedlab_engine.movie import record, save_movie

from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun import video

RIGHT = np.array([name == "RIGHT" for name in BUTTONS], dtype=np.int8)


def test_a_movie_is_rendered_by_the_engine_into_an_mp4_beside_it(tmp_path: Path):
    movie = tmp_path / "tas.movie"
    save_movie(movie, record(Console(DEMO_COLLECTION, cartridge_name("walk")), np.array([RIGHT] * 30)))

    rendered = video.render_video(load_config(write_config(tmp_path)).paths, [movie])

    assert rendered == [tmp_path / "tas.mp4"]
    assert rendered[0].stat().st_size > 0


def test_two_videos_are_stacked_side_by_side(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(video.subprocess, "run", lambda command, check, capture_output, text: commands.append(command))

    stacked = video.stack_videos(tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "ab.mp4")

    assert stacked == tmp_path / "ab.mp4"
    assert commands[0][commands[0].index("-filter_complex") + 1] == "hstack=inputs=2"
