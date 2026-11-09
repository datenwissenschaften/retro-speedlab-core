import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.policy.prior import PolicyPrior
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.speedrun import search
from datenwissenschaften.speedrun.archive_store import newest
from datenwissenschaften.speedrun.archives import Archives
from datenwissenschaften.speedrun.explore import search_segment
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.prior import ActionPrior
from datenwissenschaften.speedrun.routes import finished, load_progress, save_progress
from datenwissenschaften.speedrun.runner import Outcome, Start
from datenwissenschaften.speedrun.search import (
    UNBOUNDED,
    Cell,
    Checkpoint,
    Crossing,
    Trail,
    farthest,
    holds,
    shave,
)

ORDER = ("Walk", "Goal")
GOAL_X = 6
PIT_X = -2
MOVES = (0, 1, -1)
START = Start("Walk", (0).to_bytes(2, "big", signed=True))
NO_PREFERENCES = Preferences(ActionPrior(Path("no-prior.msgpack")), PolicyPrior(Path("no-policy.npz")))


class LineRunner:
    def __init__(self) -> None:
        self.x = 0
        self.wrapper = SimpleNamespace(
            action_table=np.zeros((3, 2, 1), dtype=np.int8),
            state_machine=self,
            routes=RouteMap((Path("no-route.json"),)),
        )

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

    def here(self, state: str) -> Start:
        return Start(state, self.snapshot())

    def memory(self) -> np.ndarray:
        return np.asarray([self.x % 256], dtype=np.uint8)

    def context(self) -> tuple[str, str]:
        return (f"{self.state_name}|{self.x}", self.state_name)

    def remaining(self) -> int | None:
        return None

    def key(self, frames: int) -> tuple[object, ...]:
        return (self.state_name, self.x)


def test_the_search_finds_the_fastest_way_into_the_next_state(tmp_path: Path):
    archives = Archives(tmp_path, 2)
    crossings = search_segment(LineRunner(), START, UNBOUNDED, 7, 1.0, None, NO_PREFERENCES, archives).crossings

    assert crossings[0].end.state == "Goal"
    assert crossings[0].frames == GOAL_X
    assert crossings[0].pressed.ravel().tolist() == [1] * GOAL_X


def test_a_bound_only_accepts_crossings_faster_than_the_movie_has(tmp_path: Path):
    archives = Archives(tmp_path, 2)
    assert search_segment(LineRunner(), START, GOAL_X, 7, 0.2, None, NO_PREFERENCES, archives).crossings == []
    assert search_segment(LineRunner(), START, GOAL_X + 1, 7, 1.0, None, NO_PREFERENCES, archives).crossings != []


def test_a_search_that_cannot_leave_returns_the_point_with_the_most_reward_as_a_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(search, "CHECKPOINT_GAIN", GOAL_X - 2)
    searched = search_segment(LineRunner(), START, GOAL_X, 7, 0.2, None, NO_PREFERENCES, Archives(tmp_path, 2))

    assert searched.crossings == []
    assert searched.checkpoint.end.state == "Walk"
    assert searched.checkpoint.gain == GOAL_X - 2
    assert searched.checkpoint.pressed.ravel().tolist() == [1] * (GOAL_X - 2)


def test_a_checkpoint_follows_the_route_map_where_it_knows_the_cells():
    def cell(frames: int, remaining: int | None, score: float) -> search.Cell:
        trail = None if frames == 0 else search.Trail(None, np.ones((frames, 1), np.int8))
        return search.Cell(START, trail, frames, 0, remaining, score, 0)

    rewarded = cell(9, None, 50.0)
    closer = cell(6, 900, 1.0)
    mapped = {"start": cell(0, 1000, 0.0), "rewarded": rewarded, "closer": closer, "near": cell(3, 960, 2.0)}

    assert search.checkpoint_of(mapped).frames == closer.frames
    assert search.checkpoint_of({**mapped, "start": cell(0, None, 0.0)}).frames == closer.frames
    assert search.checkpoint_of({**mapped, "closer": cell(6, 950, 1.0)}).frames == rewarded.frames
    unmapped = {"start": cell(0, None, 0.0), "rewarded": rewarded, "plain": cell(4, None, 1.0)}
    assert search.checkpoint_of(unmapped).frames == rewarded.frames


def test_without_enough_reward_there_is_no_checkpoint(tmp_path: Path):
    archives = Archives(tmp_path, 2)
    assert search_segment(LineRunner(), START, 2, 7, 0.2, None, NO_PREFERENCES, archives).checkpoint is None
    assert farthest([None, None]) is None
    near, far = (Checkpoint(START, np.zeros((frames, 1), np.int8), gain, None) for frames, gain in ((4, 3.0), (2, 1.0)))
    assert farthest([far, None, near]) is near
    guided, closer = (Checkpoint(START, np.zeros((2, 1), np.int8), 0.0, left) for left in (900, 400))
    assert farthest([near, guided, closer]) is closer


def test_shaving_drops_every_frame_the_crossing_does_not_need():
    detour = np.array([[1], [0], [2], [1], [1], [0], [1], [1], [1], [1]], dtype=np.int8)
    crossing = Crossing(Start("Goal", b""), detour, False, ("Goal", GOAL_X))

    shaved = shave(LineRunner(), START, crossing, 5.0)

    assert shaved.pressed.ravel().tolist() == [1] * GOAL_X
    assert shaved.end.state == "Goal"


def test_cells_near_the_remembered_route_exit_are_picked_more_often():
    near, unknown = Cell(START, None, 0, 0, 0, 0.0, 0), Cell(START, None, 0, 0, None, 0.0, 0)

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


def test_newest_skips_a_map_folder_another_worker_removed_after_listing(tmp_path: Path) -> None:
    older, newer, removed = tmp_path / "older", tmp_path / "newer", tmp_path / "removed"
    older.mkdir()
    newer.mkdir()
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    assert newest([older, removed, newer]) == [newer, older]
