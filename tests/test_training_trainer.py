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
from datenwissenschaften.training.identity import TrainingIdentity, engine_version
from datenwissenschaften.ui.live import MAX_COMPLETED_EPISODES, MAX_FRAMES_PER_REQUEST, LiveFeed


class StopTraining(Exception):
    pass


def test_identity_resets_training_when_the_release_changes(tmp_path: Path, monkeypatch):
    resets = []
    monkeypatch.setattr(identity_module, "perform_model_reset", resets.append)
    context = RunContext(load_config(write_config(tmp_path)))
    database = JsonDatabase(tmp_path / "database.json")
    database.set("engine-version:FakeGame-v0", "0.1.0")

    TrainingIdentity(context, database).require_compatible(fake_environment(tmp_path, [(3, 0)]))

    assert resets[0].game == "FakeGame-v0"
    assert database.get("engine-version:FakeGame-v0") == engine_version()


def test_identity_keeps_training_for_the_same_release(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: pytest.fail("unexpected reset"))
    monkeypatch.setattr(identity_module, "version", lambda package: "2.10.99")
    context = RunContext(load_config(write_config(tmp_path)))
    database = JsonDatabase(tmp_path / "database.json")
    database.set("engine-version:FakeGame-v0", "2.10.1")
    database.set("database-fingerprint:FakeGame-v0", None)

    TrainingIdentity(context, database).require_compatible(fake_environment(tmp_path, [(3, 0)]))

    assert database.get("engine-version:FakeGame-v0") == "2.10.99"


def test_trainer_builds_laya_resumes_checkpoints_and_restarts_after_reset(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    config_path.write_text(config_path.read_text().replace("enable: false", "enable: true"), encoding="utf-8")
    env = fake_environment(tmp_path, [(3, 0)])
    agents, ui = [], []
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(trainer_module, "configure_accelerator", lambda: "cpu")
    monkeypatch.setattr(trainer_module, "make_environment", lambda wrapper, config: env)
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: None)
    monkeypatch.setattr(trainer_module, "configure_history", lambda *args, **kwargs: ui.append("history"))
    monkeypatch.setattr(trainer_module, "start_ui", lambda settings, root: ui.append(root))
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

    with pytest.raises(StopTraining):
        trainer.train()

    assert agents[0].network.checkpoint == "fake/laya"
    assert ui == ["history", trainer.context.record_root]
    assert published["run"]["game"] == "FakeGame-v0"
    assert published["environment"]["states"] == ["Survive", "Boss"]


def test_video_playback_imports_roms_and_replays_headless(monkeypatch, tmp_path: Path):
    played, imported = [], []
    monkeypatch.setattr(sys, "argv", ["playback", "--roms-dir", str(tmp_path), "--no-audio", "run.bk2"])
    monkeypatch.setattr(video_playback, "import_roms", imported.append)
    monkeypatch.setattr(video_playback, "play_movie", lambda movie, args, monitor: played.append(movie))

    video_playback.main()

    assert imported == [tmp_path]
    assert played == ["run.bk2"]


def test_live_feed_keeps_the_latest_finished_episodes_and_serves_them_in_chunks():
    feed = LiveFeed()
    assert feed.latest_episode() == {"episode": None, "summary": {}}
    for episode_id in range(1, MAX_COMPLETED_EPISODES + 2):
        for index in range(MAX_FRAMES_PER_REQUEST + 5):
            feed.record(b"jpeg", {"timesteps": index})
        feed.finish_episode(episode_id, 60.0, {"score": 1.0, "won": False, "new_best": False}, {})

    newest = MAX_COMPLETED_EPISODES + 1
    rest = feed.episode_frames(newest, MAX_FRAMES_PER_REQUEST)

    assert len(feed.episode_frames(newest, 0)) == MAX_FRAMES_PER_REQUEST
    assert [frame["status"]["timesteps"] for frame in rest] == list(
        range(MAX_FRAMES_PER_REQUEST, MAX_FRAMES_PER_REQUEST + 5)
    )
    assert rest[0]["image"] == "anBlZw=="
    assert feed.latest_episode()["episode"]["id"] == newest
    with pytest.raises(KeyError):
        feed.episode_frames(1, 0)
