import json
from pathlib import Path

import numpy as np
import pytest
from fakes import fake_environment

from datenwissenschaften.environment import demonstration
from datenwissenschaften.environment.demonstration import load_demonstrations, movie_buttons, replay

SCRIPT = [(3, 0), (3, 5), (3, 5), (3, 5), (3, 5)]
BUTTONS = np.array([[[0, 1]], [[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int8).reshape(-1, 2)


class FakeMovie:
    def __init__(self, game: str, state: bytes, frames: list[list[bool]]) -> None:
        self.game, self.state, self.frames, self.position = game, state, frames, -1

    def get_game(self) -> str:
        return self.game

    def get_state(self) -> bytes:
        return self.state

    def step(self) -> bool:
        self.position += 1
        return self.position < len(self.frames)

    def get_key(self, button: int, player: int) -> bool:
        return self.frames[self.position][button]


def _emulator(tmp_path: Path):
    emulator = fake_environment(tmp_path, SCRIPT).env.unwrapped
    emulator.initial_state, emulator.num_buttons = b"power-on", 2
    return emulator


def test_replay_labels_each_decision_with_the_nearest_move_of_the_state_it_was_made_in(tmp_path: Path):
    env = fake_environment(tmp_path, SCRIPT)

    demonstrations = replay(env, BUTTONS)

    assert [step.action for step in demonstrations["Survive"]] == [1]
    assert [step.action for step in demonstrations["Boss"]] == [0, 0]
    assert json.loads(demonstrations["Survive"][0].state) == {"lives": 3, "score": 0}
    assert demonstrations["Boss"][0].question == "Which move beats the boss?"
    np.testing.assert_array_equal(np.array(env.env.pressed), BUTTONS)


def test_loading_replays_every_movie_without_recording_it(tmp_path: Path, monkeypatch):
    env = fake_environment(tmp_path, SCRIPT)
    record_dir = env.env.unwrapped.movie_path
    (tmp_path / "demos").mkdir()
    (tmp_path / "demos" / "menu.bk2").write_bytes(b"")
    monkeypatch.setattr(demonstration, "movie_buttons", lambda path, emulator: BUTTONS)

    demonstrations = load_demonstrations(env, tmp_path / "demos")

    assert {state: len(steps) for state, steps in demonstrations.items()} == {"Survive": 1, "Boss": 2}
    assert env.env.unwrapped.movie_path == record_dir
    assert load_demonstrations(env, tmp_path / "empty") == {}


def test_movie_buttons_skip_the_reset_frame_and_require_a_power_on_movie_of_the_game(tmp_path: Path, monkeypatch):
    emulator = _emulator(tmp_path)
    frames = [[False, False], [True, False], [False, True]]
    movies = iter(
        [
            FakeMovie("FakeGame-v0", b"power-on", frames),
            FakeMovie("OtherGame-v0", b"power-on", frames),
            FakeMovie("FakeGame-v0", b"level-2", frames),
        ]
    )
    monkeypatch.setattr(demonstration.retro, "Movie", lambda path: next(movies))

    np.testing.assert_array_equal(movie_buttons(tmp_path / "run.bk2", emulator), [[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="OtherGame-v0"):
        movie_buttons(tmp_path / "run.bk2", emulator)
    with pytest.raises(ValueError, match="power-on"):
        movie_buttons(tmp_path / "run.bk2", emulator)
