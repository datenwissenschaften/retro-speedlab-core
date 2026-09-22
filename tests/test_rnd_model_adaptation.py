from collections import deque
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.rnd import model as rnd_model_module
from datenwissenschaften.rnd.model import (
    AdaptiveRecurrentRNDModel,
    AdaptiveRecurrentRNDPPO,
    _action_count,
    _AdaptiveExplorationCallback,
    _float_or_auto,
    _observation_pixels,
    _RNDRewardWrapper,
    _StopOnModelResetCallback,
    _value_or_auto,
)

_VISUAL_SHAPE = (3, 40, 40)


class _FakeGameEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "visual": gym.spaces.Box(low=0, high=255, shape=_VISUAL_SHAPE, dtype=np.uint8),
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Discrete(4)

    def __init__(self) -> None:
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        self._steps = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._steps += 1
        terminated = self._steps >= 3
        return self.observation_space.sample(), 0.1, terminated, False, {"won": False}


def _build_model(**overrides) -> AdaptiveRecurrentRNDPPO:
    venv = DummyVecEnv([_FakeGameEnv])
    kwargs = {
        "n_steps": 3,
        "batch_size": 3,
        "n_epochs": 1,
        "device": "cpu",
        "policy_kwargs": {
            "lstm_hidden_size": 8,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "net_arch": [8],
        },
        "rnd_output_size": 8,
        "rnd_anneal_steps": 100,
        **overrides,
    }
    return AdaptiveRecurrentRNDPPO("MultiInputLstmPolicy", venv, **kwargs)


def _bare_model(**attributes) -> AdaptiveRecurrentRNDPPO:
    model = object.__new__(AdaptiveRecurrentRNDPPO)
    for name, value in attributes.items():
        setattr(model, name, value)
    return model


def test_constructor_rejects_a_non_recurrent_policy():
    with pytest.raises(ValueError, match="MultiInputLstmPolicy"):
        AdaptiveRecurrentRNDPPO("MlpPolicy", None)


def test_setup_model_rejects_non_dict_observation_spaces(monkeypatch):
    # SB3's own policy construction already rejects a non-Dict observation
    # space before this check would run, so exercise `_setup_model` directly
    # with `RecurrentPPO._setup_model` stubbed out.
    monkeypatch.setattr(RecurrentPPO, "_setup_model", lambda self: None)
    model = _bare_model(observation_space=gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32))

    with pytest.raises(ValueError, match="dictionary observations"):
        model._setup_model()


def test_learn_restarts_with_a_fresh_model_when_a_reset_is_requested(monkeypatch):
    model = _build_model()
    reset_request = object()
    consume_calls = {"count": 0}

    def fake_consume_model_reset():
        consume_calls["count"] += 1
        return reset_request if consume_calls["count"] == 1 else None

    perform_calls = []
    monkeypatch.setattr(rnd_model_module, "consume_model_reset", fake_consume_model_reset)
    monkeypatch.setattr(rnd_model_module, "perform_model_reset", perform_calls.append)

    model.learn(total_timesteps=3)

    assert perform_calls == [reset_request]
    assert consume_calls["count"] == 2


def test_learn_raises_when_no_environment_is_attached_for_a_restart(monkeypatch):
    model = _build_model()
    monkeypatch.setattr(RecurrentPPO, "learn", lambda self, **kwargs: "result")
    # Unwrap so `self.env` is a plain VecEnv rather than an `_RNDRewardWrapper`,
    # mirroring how a model can end up without the reward wrapper attached.
    model.env = model.env.venv
    monkeypatch.setattr(rnd_model_module, "consume_model_reset", lambda: object())

    with pytest.raises(RuntimeError, match="restart"):
        model.learn(total_timesteps=3)


def test_set_env_wraps_a_plain_env_and_reattaches_rnd(monkeypatch):
    model = _build_model()
    attach_calls = []
    monkeypatch.setattr(AdaptiveRecurrentRNDPPO, "_attach_rnd_to_env", lambda self: attach_calls.append(True))
    plain_venv = model.env.venv

    model.set_env(plain_venv)

    assert isinstance(model.env, _RNDRewardWrapper)
    assert attach_calls == [True]


def test_set_env_keeps_an_already_wrapped_env(monkeypatch):
    model = _build_model()
    wrapped = model.env
    monkeypatch.setattr(AdaptiveRecurrentRNDPPO, "_attach_rnd_to_env", lambda self: None)

    model.set_env(wrapped)

    assert model.env is wrapped


def test_learn_forwards_a_single_callback_alongside_the_internal_ones(monkeypatch):
    model = _build_model()
    seen_callbacks = []

    def fake_learn(self, **kwargs):
        seen_callbacks.append(kwargs["callback"])
        return "result"

    monkeypatch.setattr(RecurrentPPO, "learn", fake_learn)
    monkeypatch.setattr(rnd_model_module, "consume_model_reset", lambda: None)
    single_callback = object()

    result = model.learn(total_timesteps=3, callback=single_callback)

    assert result == "result"
    assert seen_callbacks[0][-1] is single_callback


def test_learn_forwards_a_list_of_callbacks_alongside_the_internal_ones(monkeypatch):
    model = _build_model()
    seen_callbacks = []

    def fake_learn(self, **kwargs):
        seen_callbacks.append(kwargs["callback"])
        return "result"

    monkeypatch.setattr(RecurrentPPO, "learn", fake_learn)
    monkeypatch.setattr(rnd_model_module, "consume_model_reset", lambda: None)
    extra_callback = object()

    model.learn(total_timesteps=3, callback=[extra_callback])

    assert seen_callbacks[0][-1] is extra_callback
    assert len(seen_callbacks[0]) == 3


def test_train_casts_multibinary_actions_to_float32(monkeypatch):
    train_calls = []
    monkeypatch.setattr(RecurrentPPO, "train", lambda self: train_calls.append(True))
    model = _bare_model(
        action_space=gym.spaces.MultiBinary(3),
        rollout_buffer=SimpleNamespace(actions=np.zeros((2, 3), dtype=np.int64)),
    )

    model.train()

    assert model.rollout_buffer.actions.dtype == np.float32
    assert train_calls == [True]


def test_train_leaves_non_multibinary_actions_untouched(monkeypatch):
    train_calls = []
    monkeypatch.setattr(RecurrentPPO, "train", lambda self: train_calls.append(True))
    actions = np.zeros((2,), dtype=np.int64)
    model = _bare_model(action_space=gym.spaces.Discrete(4), rollout_buffer=SimpleNamespace(actions=actions))

    model.train()

    assert model.rollout_buffer.actions is actions
    assert train_calls == [True]


def test_build_adaptive_recurrent_rnd_ppo_constructs_a_verbose_zero_model():
    venv = DummyVecEnv([_FakeGameEnv])

    model = rnd_model_module.build_adaptive_recurrent_rnd_ppo(
        venv,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        device="cpu",
        policy_kwargs={
            "lstm_hidden_size": 8,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "enable_critic_lstm": True,
            "net_arch": [8],
        },
        rnd_output_size=8,
        rnd_anneal_steps=100,
    )

    assert isinstance(model, AdaptiveRecurrentRNDPPO)
    assert model.verbose == 0


def test_action_count_covers_every_space_type():
    assert _action_count(gym.spaces.Discrete(5)) == 5
    assert _action_count(gym.spaces.MultiBinary(3)) == 3
    assert _action_count(gym.spaces.MultiDiscrete([2, 3])) == 5
    assert _action_count(gym.spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)) == 4
    assert _action_count(None) == 1


def test_observation_pixels_covers_dict_and_box_spaces():
    dict_space = gym.spaces.Dict({"visual": gym.spaces.Box(low=0, high=1, shape=(3, 4, 4), dtype=np.uint8)})
    assert _observation_pixels(dict_space) == 48
    assert _observation_pixels(gym.spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.float32)) == 4
    assert (
        _observation_pixels(gym.spaces.Dict({"ram": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)})) == 0
    )
    assert _observation_pixels(None) == 0


def test_value_and_float_or_auto_prefer_the_explicit_value():
    assert _value_or_auto(None, 7) == 7
    assert _value_or_auto(3, 7) == 3
    assert _float_or_auto(None, 1.5) == 1.5
    assert _float_or_auto(2.5, 1.5) == 2.5


def test_record_episode_outcome_tracks_wins_and_score_improvement():
    model = _bare_model(
        adaptive_episode_count=0,
        _recent_fitness=deque(maxlen=128),
        adaptive_autoconfigure=False,
        best_adaptation_fitness=None,
        adaptive_score_delta=0.0,
        episodes_since_score_improvement=0,
        episodes_since_win=0,
        adaptive_score_staleness_episodes=25,
        adaptive_no_win_staleness_episodes=50,
        adaptive_multiplier_min=0.5,
        adaptive_multiplier_max=4.0,
        adaptive_combined_multiplier=3.0,
        adaptive_stale_score_multiplier=2.0,
        adaptive_no_win_multiplier=2.0,
        adaptive_recovery_multiplier=1.0,
        adaptation_multiplier=1.0,
        adaptive_smoothing=0.1,
        rnd=None,
        ent_coef=0.0,
        base_ent_coef=0.01,
        base_learning_rate=0.0003,
        adaptive_learning_rate_min=1e-5,
        adaptive_learning_rate_max=1e-3,
        base_clip_range=0.2,
        adaptive_clip_range_min=0.08,
        adaptive_clip_range_max=0.25,
        lr_schedule=None,
        learning_rate=0.0003,
        clip_range=0.2,
        policy=None,
    )

    model.record_episode_outcome(fitness=10.0, won=True)

    assert model.best_adaptation_fitness == 10.0
    assert model.episodes_since_score_improvement == 0
    assert model.episodes_since_win == 0
    assert model.adaptation_reason == "progressing"

    model.record_episode_outcome(fitness=5.0, won=False)

    assert model.episodes_since_score_improvement == 1
    assert model.episodes_since_win == 1


def test_adapt_score_delta_updates_after_twelve_recent_episodes():
    model = _bare_model(adaptive_autoconfigure=True, _recent_fitness=deque(range(12), maxlen=128))

    model._adapt_score_delta_from_recent_fitness()

    assert model.adaptive_score_delta > 0


def test_adapt_score_delta_is_a_noop_with_few_samples_or_manual_configuration():
    model = _bare_model(adaptive_autoconfigure=True, _recent_fitness=deque([1.0], maxlen=128))
    model._adapt_score_delta_from_recent_fitness()
    assert not hasattr(model, "adaptive_score_delta")

    manual_model = _bare_model(adaptive_autoconfigure=False, _recent_fitness=deque(range(12), maxlen=128))
    manual_model._adapt_score_delta_from_recent_fitness()
    assert not hasattr(manual_model, "adaptive_score_delta")


def _exploration_model(*, episodes_since_score_improvement, episodes_since_win, rnd=None) -> AdaptiveRecurrentRNDPPO:
    return _bare_model(
        episodes_since_score_improvement=episodes_since_score_improvement,
        episodes_since_win=episodes_since_win,
        adaptive_score_staleness_episodes=5,
        adaptive_no_win_staleness_episodes=5,
        adaptive_combined_multiplier=3.0,
        adaptive_stale_score_multiplier=2.0,
        adaptive_no_win_multiplier=2.5,
        adaptive_recovery_multiplier=1.0,
        adaptive_multiplier_min=0.5,
        adaptive_multiplier_max=4.0,
        adaptation_multiplier=1.0,
        adaptive_smoothing=1.0,
        rnd=rnd,
        adaptive_rnd_update_max=1.0,
        base_rnd_update_proportion=0.25,
        ent_coef=0.0,
        base_ent_coef=0.01,
        base_learning_rate=0.0003,
        adaptive_learning_rate_min=1e-5,
        adaptive_learning_rate_max=1e-3,
        base_clip_range=0.2,
        adaptive_clip_range_min=0.08,
        adaptive_clip_range_max=0.25,
        lr_schedule=None,
        learning_rate=0.0003,
        clip_range=0.2,
        policy=None,
    )


@pytest.mark.parametrize(
    "since_score, since_win, reason",
    [
        (10, 10, "stale_score_and_no_wins"),
        (10, 0, "stale_score"),
        (0, 10, "no_wins"),
        (0, 0, "progressing"),
    ],
)
def test_adapt_exploration_picks_the_matching_reason(since_score, since_win, reason):
    model = _exploration_model(episodes_since_score_improvement=since_score, episodes_since_win=since_win)

    model._adapt_exploration()

    assert model.adaptation_reason == reason


def test_adapt_exploration_updates_the_attached_rnd_module():
    rnd = SimpleNamespace(set_adaptation_multiplier=lambda value: None, update_proportion=0.25)
    model = _exploration_model(episodes_since_score_improvement=10, episodes_since_win=10, rnd=rnd)

    model._adapt_exploration()

    assert model.adaptive_rnd_update_proportion <= model.adaptive_rnd_update_max


def test_adaptive_clip_range_returns_a_callable_schedule_unchanged():
    schedule = lambda progress: 0.3  # noqa: E731
    model = _bare_model(base_clip_range=schedule, adaptation_multiplier=1.0)

    assert model._adaptive_clip_range() is schedule


def test_adaptive_clip_range_scales_a_fixed_value_within_bounds():
    model = _bare_model(
        base_clip_range=0.4, adaptation_multiplier=4.0, adaptive_clip_range_min=0.05, adaptive_clip_range_max=0.5
    )

    assert 0.05 <= model._adaptive_clip_range() <= 0.5


def test_set_optimizer_learning_rate_updates_attached_policy_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    model = _bare_model(policy=SimpleNamespace(optimizer=optimizer))

    model._set_optimizer_learning_rate(0.5)

    assert model.learning_rate == 0.5
    assert optimizer.param_groups[0]["lr"] == 0.5


def test_set_optimizer_learning_rate_tolerates_a_missing_policy():
    model = _bare_model(policy=None)

    model._set_optimizer_learning_rate(0.25)

    assert model.learning_rate == 0.25


def test_finalize_manual_adaptation_defaults_used_when_autoconfigure_disabled():
    model = _build_model(adaptive_autoconfigure=False)

    assert model.adaptive_score_staleness_episodes == 25
    assert model.adaptive_no_win_staleness_episodes == 50


def test_cleanup_incompatible_artifacts_removes_files_symlinks_and_directories(tmp_path: Path):
    game_dir = tmp_path / "Game"
    game_dir.mkdir()
    legacy_dir = game_dir / "datenwissenschaften"
    legacy_dir.mkdir()
    (legacy_dir / "artifact").write_text("stale", encoding="utf-8")
    config = SimpleNamespace(paths=SimpleNamespace(models_dir=tmp_path), training=SimpleNamespace(game_identity="Game"))

    AdaptiveRecurrentRNDModel.cleanup_incompatible_artifacts(config)

    assert not legacy_dir.exists()


def test_cleanup_incompatible_artifacts_removes_a_stray_file(tmp_path: Path):
    game_dir = tmp_path / "Game"
    game_dir.mkdir()
    legacy_file = game_dir / "datenwissenschaften"
    legacy_file.write_text("stale", encoding="utf-8")
    config = SimpleNamespace(paths=SimpleNamespace(models_dir=tmp_path), training=SimpleNamespace(game_identity="Game"))

    AdaptiveRecurrentRNDModel.cleanup_incompatible_artifacts(config)

    assert not legacy_file.exists()


def test_cleanup_incompatible_artifacts_is_a_noop_when_nothing_to_remove(tmp_path: Path):
    config = SimpleNamespace(paths=SimpleNamespace(models_dir=tmp_path), training=SimpleNamespace(game_identity="Game"))

    AdaptiveRecurrentRNDModel.cleanup_incompatible_artifacts(config)


def test_adaptive_recurrent_rnd_model_load_and_call_delegate(monkeypatch):
    load_calls = []
    monkeypatch.setattr(AdaptiveRecurrentRNDPPO, "load", staticmethod(lambda path, **kwargs: load_calls.append(path)))
    AdaptiveRecurrentRNDModel.load("model-path", device="cpu")
    assert load_calls == ["model-path"]

    built = []
    monkeypatch.setattr(rnd_model_module, "build_adaptive_recurrent_rnd_ppo", lambda env: built.append(env))
    AdaptiveRecurrentRNDModel()("venv")
    assert built == ["venv"]


def test_stop_on_model_reset_callback_reports_whether_a_reset_was_requested(monkeypatch):
    monkeypatch.setattr(rnd_model_module, "model_reset_requested", lambda: True)
    callback = _StopOnModelResetCallback()

    assert callback._on_step() is False


def test_adaptive_exploration_callback_records_completed_episodes():
    outcomes = []
    fake_model = SimpleNamespace(record_episode_outcome=lambda *, fitness, won: outcomes.append((fitness, won)))
    callback = _AdaptiveExplorationCallback(fake_model)
    callback.model = SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=2))
    callback._on_training_start()
    callback.locals = {
        "rewards": np.asarray([1.0, 2.0]),
        "dones": np.asarray([False, True]),
        "infos": [{}, {"episode": {"r": 9.0}, "won": True}],
    }

    assert callback._on_step() is True

    assert outcomes == [(9.0, True)]
    assert callback._episode_fitness[1] == 0.0


def test_adaptive_exploration_callback_grows_its_fitness_tracker_to_match_new_environments():
    outcomes = []
    fake_model = SimpleNamespace(record_episode_outcome=lambda *, fitness, won: outcomes.append((fitness, won)))
    callback = _AdaptiveExplorationCallback(fake_model)
    callback._episode_fitness = [0.0]
    callback.locals = {
        "rewards": np.asarray([1.0, 2.0]),
        "dones": np.asarray([False, False]),
        "infos": [{}, {}],
    }

    assert callback._on_step() is True
    assert callback._episode_fitness == [1.0, 2.0]


def test_adaptive_exploration_callback_is_a_noop_without_locals():
    callback = _AdaptiveExplorationCallback(SimpleNamespace())
    callback.model = SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=1))
    callback._on_training_start()
    callback.locals = {}

    assert callback._on_step() is True


def test_rnd_reward_wrapper_reset_clears_returns_and_delegates():
    reset_calls = []
    rnd = SimpleNamespace(reset_returns=lambda: reset_calls.append(True))
    wrapper = object.__new__(_RNDRewardWrapper)
    wrapper.rnd = rnd
    wrapper.venv = SimpleNamespace(reset=lambda: "reset-result")

    assert wrapper.reset() == "reset-result"
    assert reset_calls == [True]


def test_rnd_reward_wrapper_step_wait_passes_through_when_disabled():
    wrapper = object.__new__(_RNDRewardWrapper)
    wrapper.rnd = None
    wrapper.enabled = False
    wrapper.venv = SimpleNamespace(step_wait=lambda: ("obs", "rewards", "dones", "infos"))

    assert wrapper.step_wait() == ("obs", "rewards", "dones", "infos")


def test_rnd_reward_wrapper_step_wait_requires_dict_observations():
    wrapper = object.__new__(_RNDRewardWrapper)
    wrapper.rnd = SimpleNamespace()
    wrapper.enabled = True
    wrapper.venv = SimpleNamespace(step_wait=lambda: ("not-a-dict", np.asarray([1.0]), np.asarray([False]), [{}]))

    with pytest.raises(TypeError, match="dictionary observations"):
        wrapper.step_wait()


def test_rnd_reward_wrapper_step_wait_combines_intrinsic_reward_and_uses_terminal_observations_for_the_bonus():
    seen_reward_observations = {}

    def fake_intrinsic_reward(reward_observations, dones):
        seen_reward_observations.update(reward_observations)
        return np.asarray([2.0, 3.0]), 0.5

    wrapper = object.__new__(_RNDRewardWrapper)
    wrapper.rnd = SimpleNamespace(intrinsic_reward=fake_intrinsic_reward)
    wrapper.enabled = True
    observations = {"visual": np.zeros((2, 1), dtype=np.float32)}
    terminal_observation = {"visual": np.asarray([9.0])}
    wrapper.venv = SimpleNamespace(
        step_wait=lambda: (
            observations,
            np.asarray([1.0, 1.0], dtype=np.float32),
            np.asarray([False, True]),
            [{}, {"terminal_observation": terminal_observation}],
        )
    )

    result_observations, combined_rewards, dones, infos = wrapper.step_wait()

    assert combined_rewards.tolist() == [1.0 + 0.5 * 2.0, 1.0 + 0.5 * 3.0]
    assert infos[1]["intrinsic_reward"] == 3.0
    assert infos[1]["intrinsic_reward_coefficient"] == 0.5
    # The terminal observation is substituted only for the intrinsic-reward
    # computation, not in the observations returned to the training loop.
    assert result_observations is observations
    assert seen_reward_observations["visual"][1][0] == 9.0
    assert result_observations["visual"][1][0] == 0.0
