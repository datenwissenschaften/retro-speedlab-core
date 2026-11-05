import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from datenwissenschaften.reference.trace import route_rows
from datenwissenschaften.reference.trace import write_route as write_steps
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.tas.archives import Archive
from datenwissenschaften.tas.exploration import Exploration
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.preferences import Preferences
from datenwissenschaften.tas.runner import SegmentRunner, Start, route_cell
from datenwissenschaften.tas.search import Cell


def at(x: int, y: int) -> Position:
    return Position(1, x, y, 0, False)


def write_route(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_a_cell_on_the_route_tells_how_many_frames_are_left(tmp_path: Path):
    write_route(tmp_path / "reference" / "route.json", [[1, 0, 0, 0, 2], [1, 1, 0, 1, 1]])
    routes = RouteMap((tmp_path / "reference" / "route.json",))

    assert routes.remaining(at(3, 5)) == 2
    assert routes.remaining(at(17, 0)) == 1
    assert routes.remaining(at(40, 0)) is None
    assert routes.remaining(None) is None


def test_without_a_route_file_there_is_no_route(tmp_path: Path):
    assert RouteMap((tmp_path / "route.json", tmp_path / "tas_route.json")).remaining(at(0, 0)) is None


def test_the_practice_and_tas_maps_merge_and_the_fewer_frames_left_win(tmp_path: Path):
    practice, tas = tmp_path / "reference" / "route.json", tmp_path / "reference" / "tas_route.json"
    write_route(practice, [[1, 0, 0, 0, 9], [1, 1, 0, 0, 4]])
    tas.write_text(json.dumps([[1, 0, 0, 0, 3], [1, 1, 0, 0, 8], [1, 2, 0, 0, 6]]), encoding="utf-8")
    routes = RouteMap((practice, tas))

    assert [routes.remaining(at(x, 0)) for x in (0, 16, 32, 48)] == [3, 4, 6, None]
    practice.unlink()
    assert [routes.remaining(at(x, 0)) for x in (0, 16, 32, 48)] == [3, 8, 6, None]


def test_the_tas_route_counts_the_frames_left_until_its_level_ends_and_keeps_the_fastest(tmp_path: Path):
    steps = [(0, 1, 0, 0), (1, 1, 16, 0), (2, 1, 0, 0), (3, 1, 32, 0), (200, 2, 0, 0), (201, 2, 16, 0)]
    write_steps(steps, route_rows(steps), tmp_path / "reference" / "route.json", tmp_path / "reference" / "trace.json")
    routes = RouteMap((tmp_path / "reference" / "route.json",))

    assert [routes.remaining(at(x, 0)) for x in (0, 16, 32)] == [1, 2, 0]
    assert routes.remaining(Position(2, 0, 0, 0, False)) == 1


def test_a_positioned_cell_key_names_its_route_cell_and_other_keys_none():
    assert route_cell(("Play", 1, 5, 3, 0, False, (2, 1))) == (1, 2, 1)
    assert route_cell(("Play", 2, 4, 1, 1, 0, 7)) is None
    assert route_cell(("Play", 12)) is None


def test_archived_cells_take_their_distance_from_the_newest_route_map(tmp_path: Path):
    path = tmp_path / "reference" / "route.json"
    write_route(path, [[1, 0, 0, 0, 5]])
    runner = SimpleNamespace(wrapper=SimpleNamespace(routes=RouteMap((path,))), restore=lambda start: None)
    key = ("Play", 1, 1, 1, 0, False, ())
    start = Start("Play", b"")
    archive = Archive({key: Cell(start, None, 0, 0, None, 0.0, 0)}, cast(RamNovelty, None))
    exploration = Exploration(cast(SegmentRunner, runner), start, 100, 0.0, None, cast(Preferences, None), archive)

    assert [cell.remaining for cell in exploration.cells()] == [5]
    path.write_text(json.dumps([[1, 0, 0, 0, 2]]), encoding="utf-8")
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 1_000_000_000))
    assert [cell.remaining for cell in exploration.cells()] == [2]
