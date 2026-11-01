import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeRam, FakeWrapper, SavingEmulator, fake_routes

from datenwissenschaften.route.position import Position
from datenwissenschaften.states.levels import level_states
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.practice import (
    LevelLoader,
    ensure_loader,
    level_start,
    loader_in,
    practised,
    write_practice_route,
)
from datenwissenschaften.tas.routes import Leg, Part, PartProgress, Route, Routes
from datenwissenschaften.tas.runner import SegmentRunner, Start

SCRIPT = [(3, 0), (3, 1), (3, 5), (3, 6), (3, 6)]
LEVEL_ADDRESS = 0xAA


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
    level_address = LEVEL_ADDRESS
    level_count = 2


def level_runner(tmp_path: Path) -> tuple[SegmentRunner, list[tuple[int, int]]]:
    emulator = SavingEmulator(tmp_path / "recordings", SCRIPT)
    pokes: list[tuple[int, int]] = []
    emulator.data.memory = SimpleNamespace(assign=lambda address, kind, value: pokes.append((address, value)))
    return SegmentRunner(LevelWrapper(emulator, fake_routes(tmp_path))), pokes


def test_the_frame_before_a_route_changes_the_level_loads_any_level(tmp_path: Path):
    runner, pokes = level_runner(tmp_path)
    route = Route(bytes([0]), [Segment("Survive", [[1, 0]] * 4, b"", False)])

    loader = loader_in(runner, route)

    assert loader == LevelLoader(bytes([1]), 1)
    assert level_start(runner, loader, 1, "Level2") == Start("Level2", bytes([2]))
    assert pokes == [(LEVEL_ADDRESS, 0)]
    with pytest.raises(RuntimeError, match="level 4 into no play"):
        level_start(runner, loader, 4, "Level5")


def test_the_loader_is_found_once_and_kept(tmp_path: Path):
    runner, _ = level_runner(tmp_path)
    routes = SimpleNamespace(routes={"old": Route(bytes([0]), [Segment("Survive", [[1, 0]] * 4, b"", False)])})
    path = tmp_path / "loader.msgpack"

    found = ensure_loader(path, runner, routes)

    assert ensure_loader(path, runner, SimpleNamespace(routes={})) == found == LevelLoader(bytes([1]), 1)
    assert ensure_loader(tmp_path / "none.msgpack", runner, SimpleNamespace(routes={})) is None


def test_a_beaten_part_tells_the_route_map_how_many_frames_each_place_is_from_its_end(tmp_path: Path):
    runner, _ = level_runner(tmp_path)
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", "Boss")])
    routes.save(
        Leg("Survive", Start("Survive", bytes([0])), "Boss"), [Segment("Boss", [[1, 0]] * 3, bytes([3]), False)]
    )
    path = tmp_path / "reference" / "route.json"

    write_practice_route(runner, routes, path, practised(runner, routes))

    assert practised(runner, routes) == {"Survive": PartProgress("Boss", 1, 3, True, False, {"lives": 3, "score": 6})}
    assert json.loads(path.read_text(encoding="utf-8")) == [[0, 0, 0, 2, 0]]


def test_an_unbeaten_part_reports_how_far_it_got_but_stays_out_of_the_route_map(tmp_path: Path):
    runner, _ = level_runner(tmp_path)
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", "Boss")])
    routes.save(
        Leg("Survive", Start("Survive", bytes([0])), "Boss"), [Segment("Survive", [[1, 0]] * 2, bytes([2]), False)]
    )
    path = tmp_path / "route.json"

    write_practice_route(runner, routes, path, practised(runner, routes))

    assert practised(runner, routes) == {
        "Survive": PartProgress("Survive", 1, 2, False, False, {"lives": 3, "score": 5})
    }
    assert json.loads(path.read_text(encoding="utf-8")) == []


def test_a_part_whose_start_is_not_where_the_part_before_ends_was_loaded(tmp_path: Path):
    runner, _ = level_runner(tmp_path)
    parts = [Part("Menu", "Survive", "Boss"), Part("Level 2", "Boss", None)]
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", parts)
    routes.save(Leg("Menu", Start("Survive", bytes([0])), "Boss"), [Segment("Boss", [[1, 0]], bytes([3]), False)])
    routes.save(Leg("Level 2", Start("Boss", bytes([1])), None), [Segment("Boss", [[1, 0]], bytes([4]), False)])

    assert [progress.loaded for progress in practised(runner, routes).values()] == [False, True]
    routes.save(Leg("Level 2", Start("Boss", bytes([3])), None), [Segment("Boss", [[1, 0]], bytes([4]), False)])
    assert practised(runner, routes)["Level 2"].loaded is False
