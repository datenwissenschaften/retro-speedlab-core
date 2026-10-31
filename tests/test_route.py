import json
from pathlib import Path

import numpy as np

from datenwissenschaften.reference.trace import rising_levels
from datenwissenschaften.reference.trace import write_route as write_steps
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap


def at(x: int, y: int) -> Position:
    return Position(1, x, y, 0, False)


def write_route(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_a_cell_on_the_route_tells_how_many_frames_are_left(tmp_path: Path):
    write_route(tmp_path / "reference" / "route.json", [[1, 0, 0, 0, 2], [1, 1, 0, 1, 1]])
    routes = RouteMap(tmp_path / "reference" / "route.json")

    assert routes.remaining(at(3, 5)) == 2
    assert routes.remaining(at(17, 0)) == 1
    assert routes.remaining(at(40, 0)) is None
    assert routes.remaining(None) is None


def test_without_a_route_file_there_is_no_route(tmp_path: Path):
    assert RouteMap(tmp_path / "route.json").remaining(at(0, 0)) is None


def test_the_tas_route_counts_the_frames_left_until_its_level_ends_and_keeps_the_fastest(tmp_path: Path):
    steps = [(0, 1, 0, 0), (1, 1, 16, 0), (2, 1, 0, 0), (3, 1, 32, 0), (200, 2, 0, 0), (201, 2, 16, 0)]
    write_steps(steps, tmp_path / "reference" / "route.json", tmp_path / "reference" / "trace.json")
    routes = RouteMap(tmp_path / "reference" / "route.json")

    assert [routes.remaining(at(x, 0)) for x in (0, 16, 32)] == [1, 2, 0]
    assert routes.remaining(Position(2, 0, 0, 0, False)) == 1


def test_the_levels_read_from_a_video_only_go_up():
    read = [0, 0, 2, 1, 1, 0, 1, 2, 2]
    log_probs = np.log(np.full((len(read), 3), 0.1))
    log_probs[np.arange(len(read)), read] = np.log(0.8)

    assert rising_levels(log_probs).tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2]
