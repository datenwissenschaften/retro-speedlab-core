import json
from pathlib import Path

from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap


def at(x: int, y: int) -> Position:
    return Position(1, x, y, 0, False)


def write_route(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_a_cell_on_the_route_tells_how_many_moves_are_left(tmp_path: Path):
    write_route(tmp_path / "Climb" / "route.json", [[1, 0, 0, 0, False, 0, 2], [1, 1, 0, 0, False, 1, 1]])
    routes = RouteMap(lambda state: tmp_path / state / "route.json")

    assert routes.remaining("Climb", at(3, 5)) == 2
    assert routes.remaining("Climb", at(9, 0)) == 1
    assert routes.remaining("Climb", at(40, 0)) is None
    assert routes.remaining("Climb", None) is None


def test_a_state_without_a_route_file_has_no_route(tmp_path: Path):
    routes = RouteMap(lambda state: tmp_path / state / "route.json")

    assert routes.remaining("Fall", at(0, 0)) is None
