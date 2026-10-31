import json
from pathlib import Path

from datenwissenschaften.reference.trace import write_route as write_steps
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap


def at(x: int, y: int) -> Position:
    return Position(1, x, y, 0, False)


def write_route(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_a_cell_on_the_route_tells_how_many_frames_are_left(tmp_path: Path):
    write_route(tmp_path / "reference" / "route.json", [[1, 0, 0, 0, False, 0, 2], [1, 1, 0, 0, False, 1, 1]])
    routes = RouteMap(tmp_path / "reference" / "route.json")

    assert routes.remaining(at(3, 5)) == 2
    assert routes.remaining(at(9, 0)) == 1
    assert routes.remaining(at(40, 0)) is None
    assert routes.remaining(None) is None


def test_without_a_route_file_there_is_no_route(tmp_path: Path):
    assert RouteMap(tmp_path / "route.json").remaining(at(0, 0)) is None


def test_the_tas_route_counts_the_frames_left_until_its_level_ends_and_keeps_the_fastest(tmp_path: Path):
    steps = [(0, 1, 0, 0, 0), (1, 1, 8, 0, 0), (2, 1, 0, 0, 0), (3, 1, 16, 0, 0), (200, 2, 0, 0, 0), (201, 2, 8, 0, 0)]
    write_steps(steps, tmp_path / "reference" / "route.json", tmp_path / "reference" / "trace.json")
    routes = RouteMap(tmp_path / "reference" / "route.json")

    assert [routes.remaining(at(x, 0)) for x in (0, 8, 16)] == [1, 2, 0]
    assert routes.remaining(Position(2, 0, 0, 0, False)) == 1
