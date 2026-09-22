from types import SimpleNamespace

from datenwissenschaften.callbacks import best_episode_callback as best_episode_callback_module
from datenwissenschaften.callbacks.best_episode_callback import BestEpisodeCallback
from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def test_on_training_start_prepares_episode_slots_for_every_env():
    callback = BestEpisodeCallback()
    callback.model = SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=3))

    callback._on_training_start()

    assert len(callback.active_episodes) == 3


def test_on_training_end_does_not_raise():
    BestEpisodeCallback()._on_training_end()


def test_on_step_returns_true_when_locals_are_incomplete():
    callback = BestEpisodeCallback()
    callback.locals = {"rewards": None, "dones": None, "infos": None}

    assert callback._on_step() is True


def test_on_step_finishes_episode_when_env_reports_done(monkeypatch):
    monkeypatch.setattr(best_episode_callback_module, "record_rollout_videos", lambda episodes, rollout: [])
    callback = BestEpisodeCallback()
    callback._ensure_episode_slots(1)
    callback.locals = {
        "rewards": [1.0],
        "dones": [True],
        "infos": [{"won": True, "extrinsic_reward": 1.0}],
    }

    result = callback._on_step()

    assert result is True
    assert callback.finished_episode_count == 1
    assert len(callback.episodes) == 1


def test_on_rollout_end_increments_rollout_count_and_clears_episodes(monkeypatch):
    recorded = []
    monkeypatch.setattr(
        best_episode_callback_module,
        "record_rollout_videos",
        lambda episodes, rollout: recorded.append((list(episodes), rollout)),
    )
    callback = BestEpisodeCallback()
    callback.episodes = [EpisodeRecord(0, 0)]

    result = callback._on_rollout_end()

    assert result is True
    assert callback.rollout_count == 1
    assert callback.episodes == []
    assert recorded[0][1] == 1
