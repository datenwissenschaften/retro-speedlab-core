from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FAKE_VERSION

from datenwissenschaften.speedrun import polish
from datenwissenschaften.speedrun.movie import Replay, Segment
from datenwissenschaften.speedrun.routes import Leg, Part, Route, Routes
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
    routes.save(Leg("Level 2", Start("Level2", bytes([1])), "Level3"), [piece("Level2", 2), piece("Level3", 3)])
    routes.save(Leg("Level 3", Start("Level3", bytes([3])), "Level4"), [piece("Level3", 4), piece("Level3", 5)])
    return routes


def test_the_searched_level_keeps_the_pieces_that_still_replay_from_the_new_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(polish, "verified_segments", replays_first(1))
    routes = saved_routes(tmp_path)

    moved = polish.rebased(None, routes, [LEVEL_3], LEVEL_3, bytes([7]))

    assert moved == {"Level 3": Route(bytes([7]), [piece("Level3", 90)], FAKE_VERSION)}


def test_a_beaten_level_that_no_longer_replays_keeps_the_old_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(polish, "verified_segments", replays_first(1))

    assert polish.rebased(None, saved_routes(tmp_path), [LEVEL_2, LEVEL_3], LEVEL_3, bytes([7])) is None


def test_every_later_level_starts_where_the_one_before_it_now_ends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(polish, "verified_segments", replays_first(2))

    moved = polish.rebased(None, saved_routes(tmp_path), [LEVEL_2, LEVEL_3], LEVEL_3, bytes([7]))

    assert moved is not None
    assert moved["Level 2"].entry == bytes([7]) and moved["Level 3"].entry == bytes([91])


class Walker:
    def __init__(self) -> None:
        self.x = 0
        self.wrapper = SimpleNamespace(state_machine=SimpleNamespace(state_name="Level1"))

    def restore(self, start: Start) -> None:
        self.x = start.emulator_state[0]

    def press(self, buttons: np.ndarray) -> None:
        self.x += int(buttons[0])

    def snapshot(self) -> bytes:
        return bytes([self.x])


def test_cut_points_run_back_from_the_exit_every_20_seconds():
    assert polish.cut_points(3000) == [1800, 600]
    assert polish.cut_points(500) == [0]


def test_the_route_up_to_a_cut_is_its_head_and_the_cut_is_the_new_start():
    whole, last = Segment("Level1", [[1]] * 3, bytes([3]), False), Segment("Level2", [[1]] * 4, bytes([7]), False)

    head, start = polish.played_until(Walker(), Start("Level1", bytes([0])), [whole, last], 5)

    assert head == [whole, Segment("Level1", [[1]] * 2, bytes([5]), False)]
    assert start == Start("Level1", bytes([5]))


def test_a_beaten_level_is_polished_only_every_third_rest_of_the_searched_one(tmp_path: Path):
    routes = saved_routes(tmp_path)
    polisher = polish.Polisher(SimpleNamespace(routes=routes), 0)

    assert [polisher.rest(LEVEL_2) for _ in range(3)] == [False, False, False]
    assert polisher.searches == 3
