from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FAKE_VERSION

from datenwissenschaften.speedrun import polish
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import Leg, Part, Routes
from datenwissenschaften.speedrun.runner import Start

LEVEL_1 = Part("Level 1", "Level1", "Level2")
LEVEL_2 = Part("Level 2", "Level2", "Level3")
LEVEL_3 = Part("Level 3", "Level3", "Level4")


def piece(state: str, end: int) -> Segment:
    return Segment(state, [[0]], bytes([end]), False)


def saved_routes(tmp_path: Path) -> Routes:
    routes = Routes(
        tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [LEVEL_1, LEVEL_2, LEVEL_3], FAKE_VERSION
    )
    routes.save(Leg("Level 2", Start("Level2", bytes([1])), "Level3"), [piece("Level2", 2), piece("Level3", 3)])
    routes.save(Leg("Level 3", Start("Level3", bytes([3])), "Level4"), [piece("Level3", 4), piece("Level3", 5)])
    return routes


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


def test_a_won_game_polishes_a_level_and_keeps_the_levels_after_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    routes = saved_routes(tmp_path)
    faster = [piece("Level3", 9)]
    events: list[tuple] = []
    monkeypatch.setattr(polish, "frames", lambda segments: 100 * len(segments))
    monkeypatch.setattr(polish.Polisher, "tail", lambda self, leg, segments, cut: (faster, None))
    monkeypatch.setattr(polish, "still_replaying", lambda runner, entry, segments: segments)
    monkeypatch.setattr(polish, "rebased", lambda runner, kept, name, current, end: events.append(current) or {})
    feed = SimpleNamespace(event=lambda *event: events.append(event))
    polisher = polish.Polisher(SimpleNamespace(routes=routes, runner=None, feed=feed), 0)

    assert polisher.polish(LEVEL_2, None)
    assert routes.segments("Level 2") == faster
    assert events == [None, ("improved", "Level3", 100, 100, None)]
