from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmulator, FakeRam, FakeWrapper, fake_environment, fake_routes, write_config

from datenwissenschaften.environment import factory
from datenwissenschaften.environment.wrapper import known_lives, lost_life, phases, state_class
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.runner import restore_emulator_state


def test_unknown_state_fails_fast(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    with pytest.raises(ValueError, match="Unknown state"):
        state_class((env.start_state_cls, *env.state_classes), "Missing")


def test_actions_must_be_button_sequences(tmp_path: Path):
    class Flat(FakeWrapper):
        action_table = np.array([[1, 0], [0, 1]], dtype=np.int8)

    with pytest.raises(ValueError, match="frames"):
        Flat(FakeEmulator([(3, 0)]), fake_routes(tmp_path))


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
    assert lost_life(3, CountedRam(2, 0))
    assert not lost_life(3, CountedRam(3, 0))
    assert not lost_life(None, CountedRam(2, 0))
    assert not lost_life(3, FakeRam(2, 0))


def test_a_life_lost_while_the_count_is_unreadable_still_counts_against_the_last_known_count():
    known = known_lives(3, FakeRam(2, 0))
    assert known == 3
    assert lost_life(known, CountedRam(2, 0))


def test_restoring_a_state_rereads_the_ram(tmp_path: Path):
    emulator = FakeEmulator([(3, 0)])
    restored = []
    emulator.em.set_state = restored.append
    emulator.data.update_ram = lambda: restored.append(b"reread")

    restore_emulator_state(emulator, b"saved")

    assert restored == [b"saved", b"reread"]


def test_factory_powers_on_the_collection_game_with_its_route_map(tmp_path: Path):
    config = load_config(write_config(tmp_path))

    env = factory.make_environment(FakeWrapper, config)

    assert (env.env.gamename, env.env.version) == ("games.walk.cartridge", "1.0.0")
    assert env.env.get_ram()[0] == 16
    assert env.routes.path == config.paths.cache_dir / "route.json"


def test_factory_refuses_a_cartridge_whose_version_the_collection_does_not_list(tmp_path: Path):
    config = load_config(write_config(tmp_path))
    game = replace(config.speedrun.game, version="2.0.0")

    with pytest.raises(RuntimeError, match="version 1.0.0, but the collection lists 2.0.0"):
        factory.make_environment(FakeWrapper, replace(config, speedrun=replace(config.speedrun, game=game)))
