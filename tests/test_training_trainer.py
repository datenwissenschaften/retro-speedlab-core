import sys
from pathlib import Path

import pytest
from fakes import FakeWrapper, fake_environment, fake_laya_load, write_config

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.settings import load_config
from datenwissenschaften.training import identity as identity_module
from datenwissenschaften.training import rotation as rotation_module
from datenwissenschaften.training import trainer as trainer_module
from datenwissenschaften.training import video_playback
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.identity import MODEL_LAYOUT, TrainingIdentity, engine_version
from datenwissenschaften.ui.live import MAX_COMPLETED_EPISODES, MAX_FRAMES_PER_REQUEST, MAX_REPLAYS, LiveFeed

LOST = {"score": 1.0, "won": False, "new_best": False, "full_run": False, "succeeded": False}


class StopTraining(Exception):
    pass


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
    monkeypatch.setattr(trainer_module, "make_environment", lambda wrapper, config, savestate: env)
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
    models.close()

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
    generation = feed.latest_episode()["generation"]
    assert feed.latest_episode() == {"generation": generation, "episode": None, "replays": [], "summary": {}}
    for episode_id in range(1, MAX_COMPLETED_EPISODES + 2):
        for index in range(MAX_FRAMES_PER_REQUEST + 5):
            feed.record(b"jpeg", {"timesteps": index})
        feed.finish_episode(episode_id, 60.0, LOST, {})

    newest = MAX_COMPLETED_EPISODES + 1
    rest = feed.episode_frames(generation, newest, MAX_FRAMES_PER_REQUEST)

    assert len(feed.episode_frames(generation, newest, 0)) == MAX_FRAMES_PER_REQUEST
    assert [frame["status"]["timesteps"] for frame in rest] == list(
        range(MAX_FRAMES_PER_REQUEST, MAX_FRAMES_PER_REQUEST + 5)
    )
    assert rest[0]["image"] == "anBlZw=="
    assert feed.latest_episode()["episode"]["id"] == newest
    with pytest.raises(KeyError):
        feed.episode_frames(generation, 1, 0)


def test_clearing_the_live_feed_starts_a_new_generation_without_old_frames():
    feed = LiveFeed()
    feed.record(b"jpeg", {"timesteps": 1})
    feed.finish_episode(1, 60.0, LOST, {})
    feed.record(b"half", {"timesteps": 2})
    old = feed.latest_episode()["generation"]

    feed.clear()
    feed.record(b"jpeg", {"timesteps": 3})
    feed.finish_episode(1, 60.0, LOST, {})

    latest = feed.latest_episode()
    assert latest["generation"] != old
    assert [frame["status"]["timesteps"] for frame in feed.episode_frames(latest["generation"], 1, 0)] == [3]
    with pytest.raises(KeyError):
        feed.episode_frames(old, 1, 0)


def test_full_and_successful_runs_stay_available_as_replays():
    feed = LiveFeed()
    generation = feed.latest_episode()["generation"]
    feed.record(b"jpeg", {"timesteps": 1})
    feed.finish_episode(1, 60.0, {**LOST, "full_run": True}, {})
    feed.record(b"jpeg", {"timesteps": 2})
    feed.finish_episode(2, 60.0, {**LOST, "succeeded": True}, {})
    for episode_id in range(3, MAX_REPLAYS + MAX_COMPLETED_EPISODES + 3):
        feed.record(b"jpeg", {"timesteps": episode_id})
        feed.finish_episode(episode_id, 60.0, LOST, {})

    assert [replay["id"] for replay in feed.latest_episode()["replays"]] == [1, 2]
    assert feed.episode_frames(generation, 1, 0)[0]["status"] == {"timesteps": 1}


def test_trainer_rotates_through_the_levels_until_a_reset(tmp_path: Path, monkeypatch):
    config_path = write_config(tmp_path)
    config_path.write_text(config_path.read_text().replace("[Level1]", "[Level1, Level2]"), encoding="utf-8")
    levels, outcomes = [], iter([None, None, "reset"])
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(trainer_module, "configure_accelerator", lambda: "cpu")
    monkeypatch.setattr(identity_module, "perform_model_reset", lambda request: None)

    def make(wrapper, config, savestate):
        levels.append(savestate)
        return fake_environment(tmp_path / savestate, [(3, 0)])

    def stop(request):
        raise StopTraining

    monkeypatch.setattr(trainer_module, "make_environment", make)
    monkeypatch.setattr(trainer_module.TrainingSession, "run", lambda session: next(outcomes))
    clock = iter(range(0, 100 * 3600, 3600))
    monkeypatch.setattr(rotation_module.time, "time", lambda: next(clock))
    monkeypatch.setattr(trainer_module, "perform_model_reset", stop)
    trainer = trainer_module.LayaTrainer(FakeWrapper, config_path)

    with pytest.raises(StopTraining):
        trainer.train()

    assert levels == ["Level1", "Level2", "Level1"]
    assert trainer.context.model_dir.name == "Level1"
