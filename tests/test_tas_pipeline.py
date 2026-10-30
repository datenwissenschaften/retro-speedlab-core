from concurrent.futures import Future
from pathlib import Path

import msgspec
import numpy as np
import pytest
from fakes import FakeEmulator, FakeWrapper, fake_routes, write_config

from datenwissenschaften.tas import footage, movie, run, searcher, workers
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.paths import TAS_MOVIE, atomic_write
from datenwissenschaften.tas.routes import FULL_RUN, ROUTES_FILE, Leg, Route
from datenwissenschaften.tas.run import BACKTRACK_SEARCHES
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Checkpoint, Crossing, search_segment, shave
from datenwissenschaften.tas.stuck import STUCK_FILE, Stuck, load_stuck

SCRIPT = [(3, 0), (3, 1), (3, 2), (3, 5), (3, 6), (3, 6), (3, 6), (3, 6)]
BOSS_FRAMES = 3
SECONDS = 0.05


class SavingEmulator(FakeEmulator):
    def __init__(self, movie_path: Path, script: list[tuple[int, int]]) -> None:
        super().__init__(movie_path, script)
        self.em.get_state = lambda: bytes([self.position])
        self.em.set_state = self._load
        self.em.step = self._advance
        self.em.set_button_mask = lambda mask, player: None

    def _load(self, state: bytes) -> None:
        self.position = state[0]

    def _advance(self) -> None:
        self.position = min(self.position + 1, len(self.script) - 1)

    def action_to_array(self, action: np.ndarray) -> list[np.ndarray]:
        return [np.asarray(action, dtype=np.uint8)]

    def reset(self) -> tuple[np.ndarray, dict]:
        observation = super().reset()
        if self.movie_path:
            Path(self.movie_path).mkdir(parents=True, exist_ok=True)
            (Path(self.movie_path) / f"{self.gamename}-Level1-{self.movie_id - 1:06d}.bk2").write_bytes(b"movie")
        return observation


class SyncPool:
    def submit(self, function, *arguments) -> Future:
        future: Future = Future()
        future.set_result(function(*arguments))
        return future

    def __enter__(self) -> "SyncPool":
        return self

    def __exit__(self, *exception: object) -> None:
        pass


def saving_environment(tmp_path: Path) -> FakeWrapper:
    emulator = SavingEmulator(tmp_path / "recordings", SCRIPT)
    return FakeWrapper(emulator, fake_routes(tmp_path))


def fake_render(paths: object, recordings: list[Path]) -> list[Path]:
    for recording in recordings:
        recording.with_suffix(".mp4").write_bytes(recording.read_bytes())
    return [recording.with_suffix(".mp4") for recording in recordings]


def fake_stack(left: Path, right: Path, stacked: Path) -> Path:
    stacked.write_bytes(left.read_bytes() + right.read_bytes())
    return stacked


def local_workers(monkeypatch, runner: SegmentRunner) -> None:
    monkeypatch.setattr(footage, "render_video", fake_render)
    monkeypatch.setattr(footage, "stack_videos", fake_stack)
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)
    monkeypatch.setattr(searcher, "explore", workers.explore)
    monkeypatch.setattr(searcher, "polish", workers.polish)


def local_tas(tmp_path: Path, monkeypatch, runner: SegmentRunner) -> run.Tas:
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    monkeypatch.setattr(run, "SegmentRunner", lambda wrapper: runner)
    local_workers(monkeypatch, runner)
    return run.tas_run("fakes:FakeWrapper", write_config(tmp_path), SyncPool())


def test_the_runner_restores_savestates_and_tells_when_a_segment_is_left(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    start = runner.power_on()

    outcomes = [runner.decide(0) for _ in range(BOSS_FRAMES)]
    runner.restore(start)

    assert [outcome.state for outcome in outcomes] == ["Survive", "Survive", "Boss"]
    assert runner.forward("Survive", "Boss") and not runner.forward("Boss", "Survive")
    assert runner.snapshot() == start.emulator_state == bytes([0])
    assert runner.key(31) == ("Survive", 1)
    assert runner.remaining() is None


def test_search_and_improve_write_a_verified_movie_and_keep_only_faster_segments(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    leg = Leg(FULL_RUN, runner.power_on(), "Boss")

    searched = tas.extend(leg, [])
    improved = tas.improve(leg, [Segment("Boss", [[1, 0]] * 6, bytes([6]), False)])
    run.publish(tas, improved)

    assert [(segment.state, len(segment.pressed)) for segment in searched] == [("Boss", BOSS_FRAMES)]
    assert run.frames(improved) == BOSS_FRAMES
    assert run.frames(run.load_progress(tas.routes.progress)) == BOSS_FRAMES
    assert (tas.output / TAS_MOVIE).read_bytes() == b"movie"
    assert (tas.output / "screen.mp4").read_bytes() == b"movie"
    assert (tas.output / "compare.mp4").read_bytes() == b"moviemovie"


def test_a_movie_that_no_longer_reaches_its_segments_is_cut_back(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    power_on = runner.power_on()

    kept = tas.verified(Leg(FULL_RUN, power_on, "Boss"), [Segment("Boss", [[1, 0]], bytes([1]), False)])

    assert kept == []
    assert movie.verified_segments(runner, power_on, [], tas.output / "again.bk2").verified == 0
    assert tas.searcher.best(power_on, 1, 1) is None


def test_verify_replays_every_part_from_its_own_start_and_rejects_one_that_breaks(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    config = write_config(tmp_path)
    good, bad = tmp_path / "good" / ROUTES_FILE, tmp_path / "bad" / ROUTES_FILE
    entry = runner.power_on().emulator_state
    for path, frames in ((good, BOSS_FRAMES), (bad, 1)):
        route = Route(entry, [Segment("Boss", [[1, 0]] * frames, bytes([frames]), False)])
        atomic_write(path, msgspec.msgpack.encode({"Survive": route}))

    run.verify("fakes:FakeWrapper", config, good)

    with pytest.raises(RuntimeError):
        run.verify("fakes:FakeWrapper", config, bad)


def test_a_segment_without_a_way_out_is_marked_stuck_and_the_way_into_it_searched_again(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    stuck: list[Stuck | None] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | None:
        stuck.append(load_stuck(tas.output / STUCK_FILE))
        return found(self, start, bound, rounds) if len(stuck) > 1 else None

    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [])

    assert run.frames(segments) == BOSS_FRAMES
    assert [marked and (marked.state, marked.searches) for marked in stuck] == [None, ("Survive", 1), None]
    assert load_stuck(tas.output / STUCK_FILE) is None
    assert [(event.kind, event.state) for event in tas.feed.journal.journal.events] == [
        ("stuck", "Survive"),
        ("found", "Boss"),
    ]


def test_workers_search_and_shave_from_a_savestate(tmp_path: Path, monkeypatch):
    env = saving_environment(tmp_path)
    monkeypatch.setattr(workers, "make_environment", lambda wrapper_cls, settings, worker: env)
    runner = workers.worker_runner.__wrapped__("fakes:FakeWrapper", write_config(tmp_path))
    start = runner.power_on()
    crossing = search_segment(runner, start, 100, 1, SECONDS, None).crossings[0]
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)

    assert (
        workers.explore("fakes:FakeWrapper", tmp_path, start, 100, 1, SECONDS, None).crossings[0].frames == BOSS_FRAMES
    )
    assert workers.polish("fakes:FakeWrapper", tmp_path, start, crossing, SECONDS) == shave(runner, start, crossing, 0)


def test_every_new_piece_is_followed_by_one_try_to_shorten_an_earlier_one(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    shortened: list[int] = []
    monkeypatch.setattr(run.Tas, "shorten", lambda self, leg, segments, index: shortened.append(index) or segments)

    tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [])

    assert shortened == [0]
    assert tas.turn == 1


def test_a_faster_piece_that_breaks_the_rest_of_the_run_is_not_taken(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    segments = [Segment("Survive", [[1, 0]] * 4, bytes([4]), False), Segment("Boss", [[1, 0]], bytes([5]), False)]
    faster = Crossing(Start("Survive", bytes([2])), np.zeros((2, 2), np.int8), False, ("Survive", 0))
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: faster)
    replayed = lambda runner, start, shorter, path: movie.Replay(1, {}, [bytes([2])])  # noqa: E731
    monkeypatch.setattr(run, "verified_segments", replayed)

    assert tas.shorten(Leg(FULL_RUN, runner.power_on(), "Boss"), segments, 0) is segments


def test_a_checkpoint_is_kept_as_a_piece_and_the_search_goes_on_from_it(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    checkpoint = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 1.0)
    calls: list[int] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        calls.append(bound)
        return checkpoint if len(calls) == 1 else found(self, start, bound, rounds)

    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [])

    assert [(segment.state, len(segment.pressed)) for segment in segments] == [("Survive", 1), ("Boss", 2)]
    assert [event.kind for event in tas.feed.journal.journal.events][:2] == ["checkpoint", "found"]


def test_the_search_goes_back_a_piece_only_after_three_failed_searches(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    piece = [Segment("Survive", [[0, 0]], bytes([1]), False)]

    kept = [tas.stuck(piece, Start("Survive", bytes([1]))) for _ in range(BACKTRACK_SEARCHES)]

    assert kept == [piece, piece, []]
    assert tas.barred == "Survive"


def test_checkpoints_in_a_row_count_the_pieces_that_stay_in_one_state():
    leg = Leg(FULL_RUN, Start("Survive", b""), "Boss")
    survive = Segment("Survive", [[0, 0]], b"", False)

    assert run.checkpoints_in_a_row(leg, []) == 0
    assert run.checkpoints_in_a_row(leg, [survive, survive]) == 2
    assert run.checkpoints_in_a_row(leg, [survive, Segment("Boss", [[0, 0]], b"", False)]) == 0


def test_no_checkpoint_comes_back_in_a_state_the_search_just_left_one_behind(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    useless = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 20.0)
    calls: list[int] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        calls.append(bound)
        return useless if len(calls) == 1 else found(self, start, bound, rounds)

    monkeypatch.setattr(searcher.Searcher, "best", best)
    tas.barred = "Survive"

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [])

    assert [segment.state for segment in segments] == ["Boss"]
    assert tas.barred is None
