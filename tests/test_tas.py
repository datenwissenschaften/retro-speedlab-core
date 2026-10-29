from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.routes import finished, load_progress, save_progress
from datenwissenschaften.tas.runner import Outcome, Start
from datenwissenschaften.tas.search import UNBOUNDED, Cell, Crossing, Trail, holds, search_segment, shave

ORDER = ("Walk", "Goal")
GOAL_X = 6
PIT_X = -2
MOVES = (0, 1, -1)
START = Start("Walk", (0).to_bytes(2, "big", signed=True))


class LineRunner:
    def __init__(self) -> None:
        self.x = 0
        self.wrapper = SimpleNamespace(action_table=np.zeros((3, 2, 1), dtype=np.int8), state_machine=self)

    @property
    def state_name(self) -> str:
        return "Goal" if self.x >= GOAL_X else "Walk"

    def restore(self, start: Start) -> None:
        self.x = int.from_bytes(start.emulator_state, "big", signed=True)

    def press(self, buttons: np.ndarray) -> tuple[bool, bool, float]:
        self.x += MOVES[int(buttons[0])]
        return self.x > PIT_X, False, float(MOVES[int(buttons[0])])

    def decide(self, action: int) -> Outcome:
        pressed = np.full((2, 1), action, dtype=np.int8)
        reward = 0.0
        for buttons in pressed:
            alive, _, frame_reward = self.press(buttons)
            reward += frame_reward
        return Outcome(pressed, self.state_name, alive, False, reward)

    def forward(self, origin: str, reached: str) -> bool:
        return ORDER.index(reached) > ORDER.index(origin)

    def snapshot(self) -> bytes:
        return self.x.to_bytes(2, "big", signed=True)

    def remaining(self) -> int | None:
        return None

    def key(self, frames: int) -> tuple[object, ...]:
        return (self.state_name, self.x)


def test_the_search_finds_the_fastest_way_into_the_next_state():
    crossings = search_segment(LineRunner(), START, UNBOUNDED, 7, 1.0, None)

    assert crossings[0].end.state == "Goal"
    assert crossings[0].frames == GOAL_X
    assert crossings[0].pressed.ravel().tolist() == [1] * GOAL_X


def test_a_bound_only_accepts_crossings_faster_than_the_movie_has():
    assert search_segment(LineRunner(), START, GOAL_X, 7, 0.2, None) == []
    assert search_segment(LineRunner(), START, GOAL_X + 1, 7, 1.0, None) != []


def test_shaving_drops_every_frame_the_crossing_does_not_need():
    detour = np.array([[1], [0], [2], [1], [1], [0], [1], [1], [1], [1]], dtype=np.int8)
    crossing = Crossing(Start("Goal", b""), detour, False, ("Goal", GOAL_X))

    shaved = shave(LineRunner(), START, crossing, 5.0)

    assert shaved.pressed.ravel().tolist() == [1] * GOAL_X
    assert shaved.end.state == "Goal"


def test_cells_near_the_remembered_route_exit_are_picked_more_often():
    near, unknown = Cell(START, None, 0, 0, 0, 0.0), Cell(START, None, 0, 0, None, 0.0)

    assert near.weight > 10 * unknown.weight


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


def idle_runner(alive: bool, after: str) -> SimpleNamespace:
    machine = SimpleNamespace(state_name="Step")

    def press(idle: np.ndarray) -> tuple[bool, bool, float]:
        machine.state_name = after
        return alive, False, 0.0

    forward = lambda origin, reached: ORDER.index(reached) > ORDER.index(origin)  # noqa: E731
    return SimpleNamespace(wrapper=SimpleNamespace(state_machine=machine), press=press, forward=forward)


def test_an_ending_holds_only_while_the_idle_player_lives_and_stays_in_it():
    idle = np.zeros(1, dtype=np.int8)

    assert holds(idle_runner(True, "Goal"), "Goal", idle)
    assert not holds(idle_runner(True, "Walk"), "Goal", idle)
    assert not holds(idle_runner(False, "Goal"), "Goal", idle)
