from pathlib import Path

import pytest

from datenwissenschaften.speedrun import polish
from datenwissenschaften.speedrun.movie import Replay, Segment
from datenwissenschaften.speedrun.routes import Leg, Part, Route, Routes
from datenwissenschaften.speedrun.runner import Start

LEVEL_1 = Part("Level 1", "Level1", "Level2")
LEVEL_2 = Part("Level 2", "Level2", "Level3")
LEVEL_3 = Part("Level 3", "Level3", "Level4")
MOVIE = Path("trial.bk2")


def piece(state: str, end: int) -> Segment:
    return Segment(state, [[0]], bytes([end]), False)


def replays_first(count: int):
    def replay(runner, start: Start, segments: list[Segment], movie: Path, catalog) -> Replay:
        verified = min(count, len(segments))
        return Replay(verified, {}, [bytes([90 + index]) for index in range(verified)])

    return replay


def saved_routes(tmp_path: Path) -> Routes:
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [LEVEL_1, LEVEL_2, LEVEL_3])
    routes.save(Leg("Level 2", Start("Level2", bytes([1])), "Level3"), [piece("Level2", 2), piece("Level3", 3)])
    routes.save(Leg("Level 3", Start("Level3", bytes([3])), "Level4"), [piece("Level3", 4), piece("Level3", 5)])
    return routes


def test_the_searched_level_keeps_the_pieces_that_still_replay_from_the_new_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(polish, "verified_segments", replays_first(1))
    routes = saved_routes(tmp_path)

    moved = polish.rebased(None, routes, [LEVEL_3], LEVEL_3, bytes([7]), MOVIE)

    assert moved == {"Level 3": Route(bytes([7]), [piece("Level3", 90)])}


def test_a_beaten_level_that_no_longer_replays_keeps_the_old_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(polish, "verified_segments", replays_first(1))

    assert polish.rebased(None, saved_routes(tmp_path), [LEVEL_2, LEVEL_3], LEVEL_3, bytes([7]), MOVIE) is None


def test_every_later_level_starts_where_the_one_before_it_now_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(polish, "verified_segments", replays_first(2))

    moved = polish.rebased(None, saved_routes(tmp_path), [LEVEL_2, LEVEL_3], LEVEL_3, bytes([7]), MOVIE)

    assert moved is not None
    assert moved["Level 2"].entry == bytes([7]) and moved["Level 3"].entry == bytes([91])
