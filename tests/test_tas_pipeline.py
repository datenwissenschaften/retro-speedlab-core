from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fakes import FakeEmulator, FakeWrapper, curriculum_run, fake_routes, write_config

from datenwissenschaften.laya.imitation import Example
from datenwissenschaften.settings import SpeedrunRecord, load_config
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.tas import movie, run, workers
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import SegmentRunner
from datenwissenschaften.tas.search import search_segment, shave
from datenwissenschaften.training import expert_hook
from datenwissenschaften.training.expert_hook import SNAPSHOT_DECISIONS, ExpertHook
from datenwissenschaften.training.learning import Learning
from datenwissenschaften.training.lessons import Lessons
from datenwissenschaften.training.own_wins import OwnWins

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

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
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
    curriculum = curriculum_run(tmp_path / "curriculum", ("Survive", "Boss"), tmp_path / "seeds")
    return FakeWrapper(emulator, curriculum, Landmarks(tmp_path / "landmarks.json"), fake_routes(tmp_path), "Level1")


def local_workers(monkeypatch, runner: SegmentRunner) -> None:
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)
    monkeypatch.setattr(run, "explore", workers.explore)
    monkeypatch.setattr(run, "polish", workers.polish)


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
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker, main: runner.wrapper)
    monkeypatch.setattr(run, "worker_pool", lambda count: SyncPool())
    monkeypatch.setattr(run, "render_video", lambda paths, recording: recording.with_suffix(".mp4"))
    monkeypatch.setattr(run, "SegmentRunner", lambda wrapper: runner)
    config = write_config(tmp_path)
    output = run.tas_dir(load_config(config))

    run.search("fakes:FakeWrapper", config, "Boss")
    searched = run.load_progress(output / run.PROGRESS_FILE)
    run.save_progress(output / run.PROGRESS_FILE, [Segment("Boss", [[1, 0]] * 6, bytes([6]), False)])
    run.improve("fakes:FakeWrapper", config)

    assert [(segment.state, len(segment.pressed)) for segment in searched] == [("Boss", BOSS_FRAMES)]
    assert run.frames(run.load_progress(output / run.PROGRESS_FILE)) == BOSS_FRAMES
    assert (output / run.TAS_MOVIE).read_bytes() == b"movie"


def test_a_movie_that_no_longer_reaches_its_segments_is_cut_back(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    pool = SyncPool()
    config = write_config(tmp_path)
    tas = run.Tas("fakes:FakeWrapper", config, load_config(config), runner, pool)
    tas.output.mkdir(parents=True)
    local_workers(monkeypatch, runner)

    kept = tas.verified([Segment("Boss", [[1, 0]], bytes([1]), False)], 0)

    assert kept == []
    assert movie.verified_segments(runner, [], tas.output / "again.bk2", 0) == 0
    assert tas.best(runner.power_on(), 1, 1) is None


def test_verify_accepts_a_movie_that_replays_and_rejects_one_that_does_not(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker, main: runner.wrapper)
    config = write_config(tmp_path)
    good, bad = tmp_path / "good.msgpack", tmp_path / "bad.msgpack"
    run.save_progress(good, [Segment("Boss", [[1, 0]] * BOSS_FRAMES, bytes([BOSS_FRAMES]), False)])
    run.save_progress(bad, [Segment("Boss", [[1, 0]], bytes([1]), False)])
    run.tas_dir(load_config(config)).mkdir(parents=True)

    run.verify("fakes:FakeWrapper", config, good)

    with pytest.raises(RuntimeError):
        run.verify("fakes:FakeWrapper", config, bad)


def test_the_speedrun_reports_level_splits_against_the_records():
    segments = [Segment("Play", [[0]] * 60, b"", False), Segment("Level2", [[0]] * 120, b"", False)]
    levels = {"Level 1": ("Menu", "Play"), "Level 2": ("Level2",)}

    report = run.report(segments, levels, "Menu", 60.0, (SpeedrunRecord("TAS", 400.0),), "Level2")

    assert report["seconds"] == 3.0
    assert report["splits"] == [{"level": "Level 1", "seconds": 3.0}]
    assert report["records"] == [{"name": "TAS", "seconds": 400.0}]
    assert (report["reached"], report["won"], report["searching"]) == ("Level2", False, "Level2")


def test_workers_search_shave_and_teach_from_a_savestate(tmp_path: Path, monkeypatch):
    env = saving_environment(tmp_path)
    monkeypatch.setattr(workers, "make_environment", lambda wrapper_cls, settings, worker, main: env)
    runner = workers.worker_runner.__wrapped__("fakes:FakeWrapper", write_config(tmp_path))
    start = runner.power_on()
    crossing = search_segment(runner, start, 100, 1, SECONDS)[0]
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)

    examples = workers.teach("fakes:FakeWrapper", tmp_path, start, 1, SECONDS)

    assert workers.explore("fakes:FakeWrapper", tmp_path, start, 100, 1, SECONDS)[0].frames == BOSS_FRAMES
    assert workers.polish("fakes:FakeWrapper", tmp_path, start, crossing, SECONDS) == shave(runner, start, crossing, 0)
    assert len(examples) == BOSS_FRAMES
    assert examples[0].question == "Which move survives?"


def test_a_failed_attempt_in_a_learning_state_becomes_a_lesson(tmp_path: Path, monkeypatch):
    env = saving_environment(tmp_path)
    env.reset()
    curriculum = env.curriculum.curriculum
    learning = Learning(curriculum, env.curriculum.drill)
    wins = OwnWins(curriculum.state_names, lambda state: tmp_path / f"{state}.wins.json", learning)
    taught = [Example("{}", "Which move survives?", 1)]
    monkeypatch.setattr(expert_hook, "teacher_pool", lambda: SyncPool())
    monkeypatch.setattr(expert_hook, "teach", lambda wrapper, config, start, seed, seconds: taught)
    hook = ExpertHook(env, Lessons({}, learning, wins), tmp_path / "config.yaml")
    step = SimpleNamespace(info={"state": "Survive"})

    for _ in range(SNAPSHOT_DECISIONS):
        hook.on_step(step)
    hook.on_episode_end(SimpleNamespace(curriculum_succeeded=False))
    hook.on_step(step)
    hook.on_update()

    assert wins.examples("Survive") == taught
    assert hook.lesson is None
