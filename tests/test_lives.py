from dataclasses import dataclass
from pathlib import Path

from fakes import FakeEmulator, FakeRam, FakeWrapper, curriculum_run, fake_routes

from datenwissenschaften.states.landmarks import Landmarks


@dataclass
class CountedRam(FakeRam):
    def remaining_lives(self) -> int:
        return self.lives


class CountedWrapper(FakeWrapper):
    ram_info_cls = CountedRam


def wrapper(tmp_path: Path, wrapper_cls: type[FakeWrapper], script: list[tuple[int, int]]) -> FakeWrapper:
    env = wrapper_cls(
        FakeEmulator(tmp_path, script),
        curriculum_run(tmp_path, ("Survive", "Boss"), tmp_path / "seeds"),
        Landmarks(tmp_path / "landmarks.json"),
        fake_routes(tmp_path),
        "Level1",
    )
    env.reset()
    return env


def test_losing_a_life_ends_the_episode(tmp_path: Path):
    env = wrapper(tmp_path, CountedWrapper, [(3, 0), (3, 1), (2, 2), (2, 3)])

    assert [env.step(0)[2] for _ in range(2)] == [False, True]


def test_games_without_a_life_count_play_on(tmp_path: Path):
    env = wrapper(tmp_path, FakeWrapper, [(3, 0), (3, 1), (2, 2), (2, 3)])

    assert [env.step(0)[2] for _ in range(2)] == [False, False]
