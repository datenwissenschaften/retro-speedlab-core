import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmulator, FakeRam, FakeWrapper, fake_environment, fake_routes, write_config

from datenwissenschaften.environment import factory
from datenwissenschaften.environment.recording import active_movie_path, recorded_buttons, restore_emulator_state
from datenwissenschaften.environment.wrapper import lost_life, phases, state_class
from datenwissenschaften.settings import load_config


def test_unknown_state_fails_fast(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    with pytest.raises(ValueError, match="Unknown state"):
        state_class((env.start_state_cls, *env.state_classes), "Missing")


def test_actions_must_be_button_sequences(tmp_path: Path):
    class Flat(FakeWrapper):
        action_table = np.array([[1, 0], [0, 1]], dtype=np.int8)

    with pytest.raises(ValueError, match="frames"):
        Flat(FakeEmulator(tmp_path, [(3, 0)]), fake_routes(tmp_path))


def test_the_wrapper_reads_the_ram_of_its_emulator(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 7)])

    assert env.read_ram() == FakeRam(3, 7)
    assert env.current_action_table() is env.action_table


def test_phases_run_from_the_start_state_without_repeats():
    assert phases(FakeWrapper) == ("Survive", "Boss")


@dataclass
class CountedRam(FakeRam):
    def remaining_lives(self) -> int:
        return self.lives


def test_only_fewer_known_lives_count_as_a_lost_life():
    assert lost_life(CountedRam(3, 0), CountedRam(2, 0))
    assert not lost_life(CountedRam(3, 0), CountedRam(3, 0))
    assert not lost_life(FakeRam(3, 0), FakeRam(2, 0))


def test_recording_is_mandatory(tmp_path: Path):
    with pytest.raises(RuntimeError, match="recording"):
        active_movie_path(FakeEmulator(tmp_path, [(3, 0)]))


def test_the_active_movie_is_named_after_game_state_and_attempt(tmp_path: Path):
    emulator = FakeEmulator(tmp_path, [(3, 0)])
    emulator.reset()

    assert active_movie_path(emulator) == str(tmp_path / "FakeGame-v0-Level1-000000.bk2")


def test_restoring_a_state_rereads_the_ram(tmp_path: Path):
    emulator = FakeEmulator(tmp_path, [(3, 0)])
    restored = []
    emulator.em.set_state = restored.append
    emulator.movie.set_state = restored.append

    restore_emulator_state(emulator, b"saved")

    assert restored == [b"saved", b"saved"]


def test_factory_starts_the_game_at_power_on_with_every_button(tmp_path: Path, monkeypatch):
    calls = {}
    monkeypatch.setattr(factory, "import_roms", lambda *paths: calls.setdefault("roms", paths))

    def make(game, state, render_mode, use_restricted_actions):
        calls.update(game=game, state=state, actions=use_restricted_actions)
        return FakeEmulator(tmp_path, [(3, 0)])

    monkeypatch.setattr(factory.retro, "make", make)
    config = load_config(write_config(tmp_path))

    env = factory.make_environment(FakeWrapper, config, 0)

    assert calls["roms"] == (config.paths.roms_path, config.paths.integrations_dir)
    assert (calls["game"], calls["state"]) == ("FakeGame-v0", factory.retro.State.NONE)
    assert calls["actions"] == factory.retro.Actions.ALL
    assert env.env.initial_state == env.env.em.get_state()
    assert re.fullmatch(r"PowerOn-\d{8}T\d{6}-0", env.env.statename)
    assert env.routes.path == config.paths.cache_dir / "reference" / "route.json"


class RecordedMovie:
    def __init__(self, frames: list[list[int]]) -> None:
        self.frames = frames
        self.position = -1

    def step(self) -> bool:
        self.position += 1
        return self.position < len(self.frames)

    def get_key(self, button: int, player: int) -> int:
        return self.frames[self.position][button]


def test_replays_skip_the_frame_the_recorder_adds_before_the_first_decision():
    buttons = recorded_buttons(RecordedMovie([[1, 1], [0, 1], [1, 0]]), 2)

    assert buttons.tolist() == [[0, 1], [1, 0]]
