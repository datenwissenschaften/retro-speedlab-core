import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FAKE_VERSION, FakeRam, FakeWrapper, SavingEmulator, fake_routes

from datenwissenschaften.route.position import Position
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.practice import practised, write_practice_route
from datenwissenschaften.speedrun.routes import Leg, Part, PartProgress, Routes
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.states.levels import level_states

SCRIPT = [(3, 0), (3, 1), (3, 5), (3, 6), (3, 6)]


def at(level: int, progress: int) -> SimpleNamespace:
    return SimpleNamespace(level=lambda: level, progress=lambda: progress)


def test_level_states_hand_over_one_level_at_a_time_and_end_on_a_warp():
    level_2, level_3 = level_states(2, 3)
    state = level_2()
    state.reset(at(1, 0))

    assert [level_2.__name__, level_3.__name__, level_3.goal] == ["Level2", "Level3", "Beat level 3"]
    assert state.step(at(1, 3)) == (3.0, False, False, None)
    assert state.step(at(2, 1)) == (0.0, False, False, level_3)
    last = level_3()
    last.reset(at(2, 0))
    assert last.step(at(4, 0))[1] is True
    assert last.step(at(3, 0)) == (0.0, False, False, None) and last._won()


def test_level_states_need_a_level():
    state = level_states(2, 2)[0]()

    with pytest.raises(RuntimeError, match="reads no level"):
        state.reset(FakeRam())


@dataclass
class LevelRam(FakeRam):
    def position(self) -> Position | None:
        return Position(0, self.score, 0, 0, False)

    def level(self) -> int | None:
        return self.score // 5


class LevelWrapper(FakeWrapper):
    ram_info_cls = LevelRam


def level_runner(tmp_path: Path) -> SegmentRunner:
    return SegmentRunner(LevelWrapper(SavingEmulator(SCRIPT), fake_routes(tmp_path)))


def test_a_beaten_part_tells_the_route_map_how_many_frames_each_place_is_from_its_end(tmp_path: Path):
    runner = level_runner(tmp_path)
    routes = Routes(
        tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", "Boss")], FAKE_VERSION
    )
    routes.save(
        Leg("Survive", Start("Survive", bytes([0])), "Boss"), [Segment("Boss", [[1, 0]] * 3, bytes([3]), False)]
    )
    path = tmp_path / "reference" / "route.json"

    write_practice_route(runner, routes, path, practised(runner, routes))

    assert practised(runner, routes) == {"Survive": PartProgress("Boss", 1, 3, True, False, {"Lives": 3, "Score": 6})}
    assert json.loads(path.read_text(encoding="utf-8")) == [[0, 0, 0, 0, 0]]


def test_an_unbeaten_part_reports_how_far_it_got_and_leaves_the_route_map_as_it_was(tmp_path: Path):
    runner = level_runner(tmp_path)
    routes = Routes(
        tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", "Boss")], FAKE_VERSION
    )
    routes.save(
        Leg("Survive", Start("Survive", bytes([0])), "Boss"), [Segment("Survive", [[1, 0]] * 2, bytes([2]), False)]
    )
    path = tmp_path / "route.json"
    path.write_text("[[0, 0, 0, 2, 0]]", encoding="utf-8")

    write_practice_route(runner, routes, path, practised(runner, routes))

    assert practised(runner, routes) == {
        "Survive": PartProgress("Survive", 1, 2, False, False, {"Lives": 3, "Score": 5})
    }
    assert json.loads(path.read_text(encoding="utf-8")) == [[0, 0, 0, 2, 0]]


def test_a_beaten_part_without_places_replaces_a_route_map_that_no_beaten_part_backs(tmp_path: Path):
    runner = level_runner(tmp_path)
    routes = Routes(
        tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Menu", "Survive", "Boss")], FAKE_VERSION
    )
    routes.save(Leg("Menu", Start("Survive", bytes([1])), "Boss"), [Segment("Boss", [[1, 0]], bytes([2]), False)])
    path = tmp_path / "route.json"
    path.write_text("[[0, 29, 14, 4538, 0]]", encoding="utf-8")

    write_practice_route(runner, routes, path, practised(runner, routes))

    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_a_part_whose_start_is_not_where_the_part_before_ends_was_loaded(tmp_path: Path):
    runner = level_runner(tmp_path)
    parts = [Part("Menu", "Survive", "Boss"), Part("Level 2", "Boss", None)]
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", parts, FAKE_VERSION)
    routes.save(Leg("Menu", Start("Survive", bytes([0])), "Boss"), [Segment("Boss", [[1, 0]], bytes([3]), False)])
    routes.save(Leg("Level 2", Start("Boss", bytes([1])), None), [Segment("Boss", [[1, 0]], bytes([4]), False)])

    assert [progress.loaded for progress in practised(runner, routes).values()] == [False, True]
    routes.save(Leg("Level 2", Start("Boss", bytes([3])), None), [Segment("Boss", [[1, 0]], bytes([4]), False)])
    assert practised(runner, routes)["Level 2"].loaded is False
