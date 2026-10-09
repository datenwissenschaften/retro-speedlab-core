import json
from dataclasses import dataclass
from pathlib import Path

from fakes import ACTIONS, FakeEmulator, FakeRam, FakeWrapper, curriculum_run, fake_routes

from datenwissenschaften.route import state_route
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.route.state_route import StateRoute
from datenwissenschaften.states.facts import Offset
from datenwissenschaften.states.landmarks import Landmarks

LEFT, RIGHT = range(2)
MOVES = tuple(ACTIONS.values())


def at(x: int, y: int) -> Position:
    return Position(1, x, y, 0, False)


def test_each_cell_keeps_the_move_of_the_fastest_win(tmp_path: Path):
    route = StateRoute(tmp_path / "route.json")

    route.record([(at(0, 0), LEFT), (at(8, 0), LEFT), (at(16, 0), LEFT)])
    route.record([(at(8, 0), RIGHT), (at(16, 0), RIGHT)])
    route.record([(at(16, 0), LEFT), (at(0, 0), LEFT), (at(16, 0), RIGHT)])

    assert route.move(at(0, 0)) == state_route.Move(LEFT, 2)
    assert route.move(at(9, 3)) == state_route.Move(RIGHT, 2)
    assert route.move(at(16, 0)) == state_route.Move(RIGHT, 1)


def test_emulators_share_one_route_file(tmp_path: Path):
    first, second = StateRoute(tmp_path / "route.json"), StateRoute(tmp_path / "route.json")

    first.record([(at(0, 0), LEFT)])
    second.record([(at(40, 0), RIGHT)])

    assert json.loads((tmp_path / "route.json").read_text()) == [[1, 0, 0, 0, False, 0, 1], [1, 5, 0, 0, False, 1, 1]]
    assert StateRoute(tmp_path / "route.json").move(at(0, 0)).action == LEFT


def test_on_the_route_laya_reads_the_move_and_off_it_the_way_back(tmp_path: Path):
    routes = RouteMap(lambda state: tmp_path / state / "route.json", MOVES)
    routes.record("Climb", [(at(0, 0), RIGHT), (at(8, 0), RIGHT)])

    assert routes.facts("Climb", at(9, 2)) == {"route": "move right"}
    assert routes.facts("Climb", at(40, -20)) == {"to_route": Offset(-28, 24)}
    assert routes.facts("Climb", Position(2, 0, 0, 0, False)) == {}
    assert routes.facts("Climb", None) == {}
    assert routes.facts("Fall", at(0, 0)) == {}
    assert routes.summary() == {"Climb": 2}


def test_a_route_another_emulator_wrote_arrives_after_the_reload_pause(monkeypatch, tmp_path: Path):
    reader, writer = StateRoute(tmp_path / "route.json"), StateRoute(tmp_path / "route.json")
    assert reader.move(at(0, 0)) is None

    writer.record([(at(0, 0), RIGHT)])
    assert reader.move(at(0, 0)) is None
    monkeypatch.setattr(state_route, "RELOAD_SECONDS", 0.0)

    assert reader.move(at(0, 0)).action == RIGHT


@dataclass
class PlacedRam(FakeRam):
    def position(self) -> Position:
        return at(self.score * 8, 0)


class PlacedWrapper(FakeWrapper):
    ram_info_cls = PlacedRam


def test_leaving_a_state_forward_teaches_its_route(tmp_path: Path):
    script = [(3, score) for score in range(7)]
    env = PlacedWrapper(
        FakeEmulator(tmp_path, script),
        curriculum_run(tmp_path, ("Survive", "Boss"), tmp_path / "seeds"),
        Landmarks(tmp_path / "landmarks.json"),
        fake_routes(tmp_path),
        "Level1",
    )
    env.reset()

    states = [env.step(RIGHT)[4]["state"] for _ in range(6)]
    reader = RouteMap(lambda state: tmp_path / "models" / state / "route.json", MOVES)

    assert states[4:] == ["Boss", "Boss"]
    assert reader.facts("Survive", at(0, 0)) == {"route": "move right"}
    assert reader.facts("Boss", at(40, 0)) == {}
    assert env.trail == [(at(40, 0), RIGHT)]


def test_a_respawn_drops_the_moves_that_led_to_the_lost_life(tmp_path: Path):
    script = [(3, 0), (3, 1), (3, 30), (3, 31)]
    env = PlacedWrapper(
        FakeEmulator(tmp_path, script),
        curriculum_run(tmp_path, ("Survive", "Boss"), tmp_path / "seeds"),
        Landmarks(tmp_path / "landmarks.json"),
        fake_routes(tmp_path),
        "Level1",
    )
    env.reset()

    for _ in range(3):
        env.step(LEFT)

    assert env.trail == [(at(240, 0), LEFT)]
