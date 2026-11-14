from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FAKE_VERSION

from datenwissenschaften.speedrun import routes as routes_module
from datenwissenschaften.speedrun.movie import Replay, Segment
from datenwissenschaften.speedrun.rebase import join, rebased
from datenwissenschaften.speedrun.routes import Leg, Part, Route, Routes
from datenwissenschaften.speedrun.run import Tas
from datenwissenschaften.speedrun.runner import Start

LEVEL_1 = Part("Level 1", "Level1", "Level2")
LEVEL_2 = Part("Level 2", "Level2", "Level3")
LEVEL_3 = Part("Level 3", "Level3", "Level4")


def piece(state: str, end: int) -> Segment:
    return Segment(state, [[0]], bytes([end]), False)


def replays_first(count: int):
    def replay(runner, start: Start, segments: list[Segment]) -> Replay:
        verified = min(count, len(segments))
        return Replay(verified, {}, [bytes([90 + index]) for index in range(verified)])

    return replay


def saved_routes(tmp_path: Path) -> Routes:
    routes = Routes(
        tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [LEVEL_1, LEVEL_2, LEVEL_3], FAKE_VERSION
    )
    routes.save(Leg("Level 1", Start("Level1", bytes([0])), "Level2"), [piece("Level2", 1)])
    routes.save(Leg("Level 2", Start("Level2", bytes([1])), "Level3"), [piece("Level2", 2), piece("Level3", 3)])
    routes.save(Leg("Level 3", Start("Level3", bytes([3])), "Level4"), [piece("Level3", 4), piece("Level4", 5)])
    return routes


def test_the_searched_level_keeps_the_pieces_that_still_replay_from_the_new_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(1))

    moved = rebased(None, saved_routes(tmp_path), "Level 2", "Level 3", bytes([7]))

    assert moved == {"Level 3": Route(bytes([7]), [piece("Level3", 90)], FAKE_VERSION)}


def test_a_beaten_level_that_no_longer_replays_keeps_the_old_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(1))

    assert rebased(None, saved_routes(tmp_path), "Level 1", "Level 3", bytes([7])) is None


def test_every_later_level_starts_where_the_one_before_it_now_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(2))

    moved = rebased(None, saved_routes(tmp_path), "Level 1", None, bytes([7]))

    assert moved is not None
    assert moved["Level 2"].entry == bytes([7]) and moved["Level 3"].entry == bytes([91])


def test_a_faster_level_moves_every_beaten_level_after_it_to_its_new_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(2))
    routes = saved_routes(tmp_path)
    tas = SimpleNamespace(runner=None, routes=routes)

    assert Tas.rebase(tas, Leg("Level 1", Start("Level1", bytes([0])), "Level2"), bytes([7]))

    stored = Routes(routes.path, routes.progress, routes.parts, FAKE_VERSION)
    assert stored.routes["Level 2"].entry == bytes([7]) and stored.routes["Level 3"].entry == bytes([91])


def test_a_faster_level_stays_out_while_a_beaten_level_after_it_breaks_from_its_new_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(1))
    routes = saved_routes(tmp_path)
    tas = SimpleNamespace(runner=None, routes=routes)

    assert not Tas.rebase(tas, Leg("Level 1", Start("Level1", bytes([0])), "Level2"), bytes([7]))

    assert routes.routes["Level 2"].entry == bytes([1])


def test_a_level_started_away_from_where_the_level_before_ends_starts_there_again_with_the_pieces_that_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(routes_module, "verified_segments", replays_first(1))
    routes = saved_routes(tmp_path)
    routes.save(Leg("Level 1", Start("Level1", bytes([0])), "Level2"), [piece("Level2", 8)])

    join(None, routes)

    assert routes.routes["Level 2"] == Route(bytes([8]), [piece("Level2", 90)], FAKE_VERSION)
    assert not routes.loaded(LEVEL_2)
    assert routes.stitched() == [piece("Level2", 8), piece("Level2", 90)]


def test_levels_that_already_start_where_the_level_before_ends_are_left_alone(tmp_path: Path):
    routes = saved_routes(tmp_path)

    join(None, routes)

    assert [route.entry for route in routes.routes.values()] == [bytes([0]), bytes([1]), bytes([3])]
    assert len(routes.stitched()) == 5
