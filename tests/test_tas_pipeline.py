from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeEmulator, FakeWrapper, fake_routes, write_config

from datenwissenschaften.settings import load_config
from datenwissenschaften.tas import movie, run, workers
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.paths import PROGRESS_FILE, TAS_MOVIE, tas_dir
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Crossing, search_segment, shave
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


def local_workers(monkeypatch, runner: SegmentRunner) -> None:
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)
    monkeypatch.setattr(run, "explore", workers.explore)
    monkeypatch.setattr(run, "polish", workers.polish)


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
    local_workers(monkeypatch, runner)
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    monkeypatch.setattr(run, "worker_pool", lambda count: SyncPool())
    monkeypatch.setattr(run, "render_video", lambda paths, recording: recording.with_suffix(".mp4"))
    monkeypatch.setattr(run, "SegmentRunner", lambda wrapper: runner)
    config = write_config(tmp_path)
    output = tas_dir(load_config(config))

    run.search("fakes:FakeWrapper", config, "Boss")
    searched = run.load_progress(output / PROGRESS_FILE)
    run.save_progress(output / PROGRESS_FILE, [Segment("Boss", [[1, 0]] * 6, bytes([6]), False)])
    run.improve("fakes:FakeWrapper", config)

    assert [(segment.state, len(segment.pressed)) for segment in searched] == [("Boss", BOSS_FRAMES)]
    assert run.frames(run.load_progress(output / PROGRESS_FILE)) == BOSS_FRAMES
    assert (output / TAS_MOVIE).read_bytes() == b"movie"


def test_a_movie_that_no_longer_reaches_its_segments_is_cut_back(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)

    kept = tas.verified([Segment("Boss", [[1, 0]], bytes([1]), False)], 0)

    assert kept == []
    assert movie.verified_segments(runner, [], tas.output / "again.bk2", 0).verified == 0
    assert tas.best(runner.power_on(), 1, 1) is None


def test_verify_accepts_a_movie_that_replays_and_rejects_one_that_does_not(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    config = write_config(tmp_path)
    good, bad = tmp_path / "good.msgpack", tmp_path / "bad.msgpack"
    run.save_progress(good, [Segment("Boss", [[1, 0]] * BOSS_FRAMES, bytes([BOSS_FRAMES]), False)])
    run.save_progress(bad, [Segment("Boss", [[1, 0]], bytes([1]), False)])

    run.verify("fakes:FakeWrapper", config, good)

    with pytest.raises(RuntimeError):
        run.verify("fakes:FakeWrapper", config, bad)


def test_a_segment_without_a_way_out_is_marked_stuck_and_the_way_into_it_searched_again(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = run.Tas.best
    stuck: list[Stuck | None] = []

    def best(self: run.Tas, start: Start, bound: int, rounds: int) -> Crossing | None:
        stuck.append(load_stuck(tas.output / STUCK_FILE))
        return found(self, start, bound, rounds) if len(stuck) > 1 else None

    monkeypatch.setattr(run.Tas, "best", best)

    segments = tas.extend([], "Boss")

    assert run.frames(segments) == BOSS_FRAMES
    assert [marked and (marked.state, marked.searches) for marked in stuck] == [None, ("Survive", 1)]
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
    crossing = search_segment(runner, start, 100, 1, SECONDS, None)[0]
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)

    assert workers.explore("fakes:FakeWrapper", tmp_path, start, 100, 1, SECONDS, None)[0].frames == BOSS_FRAMES
    assert workers.polish("fakes:FakeWrapper", tmp_path, start, crossing, SECONDS) == shave(runner, start, crossing, 0)
