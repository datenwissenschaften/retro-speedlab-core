import sys
from pathlib import Path

import pytest
from fakes import FakeWrapper, fake_environment, fake_laya_load, write_config

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.settings import load_config
from datenwissenschaften.training import identity as identity_module
from datenwissenschaften.training import trainer as trainer_module
from datenwissenschaften.training import video_playback
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.identity import MODEL_LAYOUT, TrainingIdentity, engine_version


class StopTraining(Exception):
    pass


class NoPractice:
    def __init__(self, wrapper_cls, config, speedrun: bool, workers: range) -> None:
        self.workers = workers

    def close(self) -> None:
        pass


class NoTeam:
    def __init__(self, advisors, practice, lab_run, lessons) -> None:
        self.coach = None
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_identity_resets_training_when_the_release_changes(tmp_path: Path, monkeypatch):
    resets = []
    monkeypatch.setattr(identity_module, "perform_model_reset", resets.append)
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    database = JsonDatabase(tmp_path / "database.json")
    database.set("engine-version:FakeGame-v0", "0.1.0")

    TrainingIdentity(context, database).require_compatible(fake_environment(tmp_path, [(3, 0)]))

    assert resets[0].game == "FakeGame-v0"
    assert database.get("engine-version:FakeGame-v0") == engine_version()


def test_identity_keeps_training_for_the_same_release(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: pytest.fail("unexpected reset"))
    monkeypatch.setattr(identity_module, "version", lambda package: "2.10.99")
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    database = JsonDatabase(tmp_path / "database.json")
    database.set("engine-version:FakeGame-v0", "2.10.1")
    database.set("database-fingerprint:FakeGame-v0", None)
    database.set("model-layout:FakeGame-v0", MODEL_LAYOUT)

    TrainingIdentity(context, database).require_compatible(fake_environment(tmp_path, [(3, 0)]))

    assert database.get("engine-version:FakeGame-v0") == "2.10.99"


def test_identity_resets_training_when_the_model_layout_changes(tmp_path: Path, monkeypatch):
    resets = []
    monkeypatch.setattr(identity_module, "perform_model_reset", resets.append)
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    database = JsonDatabase(tmp_path / "database.json")
    database.set("engine-version:FakeGame-v0", engine_version())
    database.set("database-fingerprint:FakeGame-v0", None)

    TrainingIdentity(context, database).require_compatible(fake_environment(tmp_path, [(3, 0)]))

    assert resets[0].game == "FakeGame-v0"
    assert database.get("model-layout:FakeGame-v0") == MODEL_LAYOUT


def test_trainer_builds_laya_resumes_checkpoints_and_restarts_after_reset(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    config_path.write_text(config_path.read_text().replace("enable: false", "enable: true"), encoding="utf-8")
    env = fake_environment(tmp_path, [(3, 0)])
    agents, ui = [], []
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(trainer_module, "configure_accelerator", lambda: "cpu")
    monkeypatch.setattr(trainer_module, "make_environment", lambda wrapper, config, worker, record: env)
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: None)
    monkeypatch.setattr(trainer_module, "PracticeEnvironments", NoPractice)
    monkeypatch.setattr(trainer_module, "AdvisorTeam", NoTeam)
    monkeypatch.setattr(trainer_module, "configure_history", lambda *args, **kwargs: ui.append("history"))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(trainer_module, "start_ui", lambda settings, root, reports, digest: ui.append(root))
    published = {}
    monkeypatch.setattr(
        trainer_module, "publish_metadata", lambda section, values, **kwargs: published.update({section: values})
    )

    def run(session):
        agents.append(session.models.agent)
        return "reset"

    def stop(request):
        raise StopTraining

    monkeypatch.setattr(trainer_module.TrainingSession, "run", run)
    monkeypatch.setattr(trainer_module, "perform_model_reset", stop)
    trainer = trainer_module.LayaTrainer(FakeWrapper, config_path)
    models = trainer._models()
    models.activate("Survive")
    models.save()
    models.close()

    with pytest.raises(StopTraining):
        trainer.train()

    assert agents[0].network.checkpoint == "fake/laya"
    assert ui == ["history", trainer.context.record_root]
    assert published["run"]["game"] == "FakeGame-v0"
    assert published["environment"]["states"] == ["Survive", "Boss"]


def test_video_playback_imports_roms_and_replays_headless(monkeypatch, tmp_path: Path):
    played, imported = [], []
    argv = ["playback", "--roms-dir", str(tmp_path), "--integrations-dir", "ints", "--no-audio", "run.bk2"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(video_playback, "import_roms", lambda *paths: imported.append(paths))
    monkeypatch.setattr(video_playback, "play_movie", lambda movie, args, monitor: played.append(movie))

    video_playback.main()

    assert imported == [(tmp_path, Path("ints"))]
    assert played == ["run.bk2"]


def test_trainer_plays_the_full_game_and_speedruns_it_once_beaten(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    sessions, outcomes, wins = [], iter([None, "reset"]), iter([0, trainer_module.BEATEN_FULL_RUN_WINS])
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(trainer_module, "configure_accelerator", lambda: "cpu")
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: None)
    monkeypatch.setattr(trainer_module, "PracticeEnvironments", NoPractice)
    monkeypatch.setattr(trainer_module, "AdvisorTeam", NoTeam)
    monkeypatch.setattr(
        trainer_module, "make_environment", lambda wrapper, config, worker, record: fake_environment(tmp_path, [(3, 0)])
    )

    def run(session):
        sessions.append(session.env.speedrun)
        return next(outcomes)

    def stop(request):
        raise StopTraining

    monkeypatch.setattr(trainer_module.TrainingSession, "run", run)
    monkeypatch.setattr(trainer_module, "level_full_run_wins", lambda level: next(wins))
    monkeypatch.setattr(trainer_module, "perform_model_reset", stop)
    trainer = trainer_module.LayaTrainer(FakeWrapper, config_path)

    with pytest.raises(StopTraining):
        trainer.train()

    assert sessions == [False, True]
    assert trainer.context.model_dir.name == "FakeGame-v0"
