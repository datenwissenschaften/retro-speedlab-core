from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.run import finished, load_progress, save_progress
from datenwissenschaften.tas.runner import Outcome, Start
from datenwissenschaften.tas.search import Trail, search_segment

ORDER = ("Walk", "Goal")
GOAL_X = 6
PIT_X = -2
UNBOUNDED = 10**9
START = Start("Walk", (0).to_bytes(2, "big", signed=True))


class LineRunner:
    def __init__(self) -> None:
        self.wrapper = SimpleNamespace(action_table=np.zeros((3, 2, 1), dtype=np.int8))
        self.x = 0

    def restore(self, start: Start) -> None:
        self.x = int.from_bytes(start.emulator_state, "big", signed=True)

    def decide(self, action: int) -> Outcome:
        self.x += (0, 1, -1)[action]
        state = "Goal" if self.x >= GOAL_X else "Walk"
        return Outcome(np.full((2, 1), action, dtype=np.int8), state, self.x > PIT_X, False)

    def forward(self, origin: str, reached: str) -> bool:
        return ORDER.index(reached) > ORDER.index(origin)

    def snapshot(self) -> bytes:
        return self.x.to_bytes(2, "big", signed=True)

    def key(self, frames: int) -> tuple[object, ...]:
        return ("Walk", self.x)


def test_the_search_finds_the_fastest_way_into_the_next_state():
    crossing = search_segment(LineRunner(), START, UNBOUNDED, 7, 1.0)

    assert crossing is not None
    assert crossing.end.state == "Goal"
    assert crossing.frames == 2 * GOAL_X
    assert crossing.pressed.ravel().tolist() == [1] * 2 * GOAL_X


def test_a_bound_only_accepts_crossings_faster_than_the_movie_has():
    assert search_segment(LineRunner(), START, 2 * GOAL_X, 7, 0.2) is None
    assert search_segment(LineRunner(), START, 2 * GOAL_X + 1, 7, 1.0) is not None


def test_a_trail_replays_its_chunks_in_order():
    trail = Trail(Trail(None, np.array([[1], [2]])), np.array([[3]]))

    assert trail.frames().ravel().tolist() == [1, 2, 3]


def test_progress_survives_a_restart_and_a_won_or_target_segment_finishes_the_run(tmp_path: Path):
    segments = [Segment("Play", [[0, 1]], b"state", False), Segment("Level2", [[1, 0]], b"later", False)]

    save_progress(tmp_path / "progress.msgpack", segments)

    assert load_progress(tmp_path / "progress.msgpack") == segments
    assert load_progress(tmp_path / "missing.msgpack") == []
    assert finished(segments, "Level2")
    assert not finished(segments, "Level3")
    assert finished([Segment("Door5", [], b"", True)], "Level3")
    assert not finished([], "Level2")
