from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften import state_trainer as state_trainer_module
from datenwissenschaften.state_trainer import SavestateScheduler, StateTrainer

_VISUAL_SHAPE = (3, 40, 40)


class _StopLoop(Exception):
    pass


class _FakeStateEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {
            "visual": gym.spaces.Box(low=0, high=255, shape=_VISUAL_SHAPE, dtype=np.uint8),
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.step_count = 0
        self.raise_after_steps: int | None = None
        self.segment_end_after_step = 1
        self.current_state = "Find"
        self.initial_savestates: list[str] = []
        self.reset_training_memory_calls = 0

    def reset(self, *, seed=None, options=None):
        return self.observation_space.sample(), {}

    def step(self, action):
        self.step_count += 1
        if self.raise_after_steps is not None and self.step_count > self.raise_after_steps:
            raise _StopLoop
        segment_end = self.step_count >= self.segment_end_after_step
        info = (
            {
                "state_segment_end": True,
                "state_return": 1.0,
                "won": True,
                "started_from_initial_savestate": True,
            }
            if segment_end
            else {"state_segment_end": False}
        )
        return self.observation_space.sample(), 1.0, False, False, info

    def training_state_names(self):
        return ["Find"]

    def set_terminate_on_transition(self, value):
        pass

    def set_transition_bonus(self, value):
        pass

    def state_name(self):
        return self.current_state

    def set_initial_savestate(self, savestate):
        self.initial_savestates.append(savestate)

    def curriculum_progress(self):
        return {"Find": {"wins": 1}}

    def reset_training_memory(self):
        self.reset_training_memory_calls += 1


class _FakeEpisodeCallback:
    instances: list["_FakeEpisodeCallback"] = []

    def __init__(self, *args, **kwargs):
        self.training_end_calls = 0
        _FakeEpisodeCallback.instances.append(self)

    def init_callback(self, model):
        pass

    def on_training_start(self, locals_, globals_):
        pass

    def update_locals(self, locals_):
        pass

    def on_step(self):
        pass

    def on_rollout_end(self):
        pass

    def on_training_end(self):
        self.training_end_calls += 1


class _FakeRuntimeTrainer:
    instances: list["_FakeRuntimeTrainer"] = []

    def __init__(self, *, config_path, state_name):
        self.config_path = config_path
        self.state_name = state_name
        self.savestate_calls: list[str] = []
        self.configure_runtime_calls = 0
        _FakeRuntimeTrainer.instances.append(self)

    def _set_savestate(self, savestate):
        self.savestate_calls.append(savestate)

    def _configure_runtime(self):
        self.configure_runtime_calls += 1

    _environment_metadata = staticmethod(lambda venv: {"class": "Fake"})


class FakeRedisStore:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.values: dict[tuple, object] = {}
        self.deleted_prefixes: list[tuple] = []
        self.deleted: list[tuple] = []

    def get(self, *parts, default=None):
        return self.values.get(parts, default)

    def set(self, *parts, value):
        self.values[parts] = value

    def delete_prefix(self, *parts):
        self.deleted_prefixes.append(parts)

    def delete(self, *parts):
        self.deleted.append(parts)


def _config(tmp_path: Path, *, savestates, ui_enabled: bool):
    return SimpleNamespace(
        paths=SimpleNamespace(
            models_dir=tmp_path / "models", record_dir=tmp_path / "recordings", cache_dir=tmp_path / "cache"
        ),
        training=SimpleNamespace(
            game="Game",
            game_identity="Game",
            active_savestate=savestates[0] if savestates else None,
            savestates=savestates,
            num_envs=1,
        ),
        ui=SimpleNamespace(enabled=ui_enabled, redis_url="redis://example", history_key_prefix="prefix"),
        upload=SimpleNamespace(url="https://example.test", api_key=None),
    )


def _model_builder():
    built: list[tuple[str, object]] = []

    class _ModelBuilder:
        def build(self, venv, *, state_name):
            model = RecurrentPPO(
                "MultiInputLstmPolicy",
                venv,
                n_steps=1,
                batch_size=1,
                n_epochs=1,
                device="cpu",
                policy_kwargs={
                    "lstm_hidden_size": 8,
                    "n_lstm_layers": 1,
                    "shared_lstm": False,
                    "enable_critic_lstm": True,
                    "net_arch": [8],
                },
            )
            built.append((state_name, model))
            return model

    return _ModelBuilder(), built


def _patch_collaborators(monkeypatch, *, config):
    monkeypatch.setattr(state_trainer_module, "load_config", lambda path: config)
    monkeypatch.setattr(state_trainer_module, "RedisStore", FakeRedisStore)
    monkeypatch.setattr(state_trainer_module, "BestEpisodeCallback", _FakeEpisodeCallback)
    monkeypatch.setattr(state_trainer_module, "EpisodeTelemetryCallback", _FakeEpisodeCallback)
    monkeypatch.setattr(state_trainer_module, "UploadEpisodeCallback", _FakeEpisodeCallback)
    monkeypatch.setattr(state_trainer_module, "Trainer", _FakeRuntimeTrainer)
    _FakeEpisodeCallback.instances = []
    _FakeRuntimeTrainer.instances = []


def test_train_validates_and_builds_a_model_per_configured_state(monkeypatch):
    trainer = StateTrainer(
        model_builder=SimpleNamespace(build=lambda venv, *, state_name: f"model-{state_name}"), transition_bonus=0.5
    )
    calls = []
    monkeypatch.setattr(
        StateTrainer, "_train_segmented", lambda self, venv, models: calls.append((venv, models)) or models
    )
    env_calls = []
    venv = SimpleNamespace(
        env_method=lambda name, *args: (
            env_calls.append((name, args)) or ([["Find", "Eat"]] if name == "training_state_names" else None)
        )
    )

    result = trainer.train(venv)

    assert result == {"Find": "model-Find", "Eat": "model-Eat"}
    assert ("set_transition_bonus", (0.5,)) in env_calls
    assert calls[0][1] == result


def test_train_rejects_environments_with_no_configured_states():
    trainer = StateTrainer(model_builder=SimpleNamespace(build=lambda venv, *, state_name: None), transition_bonus=0.0)
    venv = SimpleNamespace(env_method=lambda name, *args: [[]])

    with pytest.raises(ValueError, match="No training states"):
        trainer.train(venv)


def test_train_segmented_runs_one_full_update_cycle_without_ui(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert env.step_count == 2
    assert _FakeRuntimeTrainer.instances[0].configure_runtime_calls == 1
    assert all(instance.training_end_calls == 0 for instance in _FakeEpisodeCallback.instances)


def test_train_segmented_rotates_savestates_after_a_completed_run(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1", "Level2"), ui_enabled=True)
    _patch_collaborators(monkeypatch, config=config)
    monkeypatch.setattr(state_trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: None)
    monkeypatch.setattr(state_trainer_module, "configure_training_control", lambda **kwargs: None)
    monkeypatch.setattr(state_trainer_module, "start_ui", lambda settings: None)
    published = []
    monkeypatch.setattr(
        state_trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append(section),
    )
    runtime_savestate_calls = []
    monkeypatch.setattr(
        state_trainer_module,
        "get_runtime",
        lambda: SimpleNamespace(set_savestate=runtime_savestate_calls.append),
    )
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert runtime_savestate_calls == ["Level2"]
    assert env.initial_savestates == ["Level1", "Level2"]
    assert env.reset_training_memory_calls == 0
    assert "savestate_curriculum" in published


def test_train_segmented_publishes_periodic_progress_when_ui_enabled_without_rotation(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=True)
    _patch_collaborators(monkeypatch, config=config)
    monkeypatch.setattr(state_trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: None)
    monkeypatch.setattr(state_trainer_module, "configure_training_control", lambda **kwargs: None)
    monkeypatch.setattr(state_trainer_module, "start_ui", lambda settings: None)
    published = []
    monkeypatch.setattr(
        state_trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append(section),
    )
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert "state_training" in published


def test_train_segmented_warns_about_unsupported_additional_callbacks(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    trainer = StateTrainer(
        model_builder=builder,
        transition_bonus=0.0,
        additional_callbacks=[SimpleNamespace()],
        config_path=tmp_path / "config.yaml",
    )
    warnings = []
    monkeypatch.setattr(state_trainer_module.logger, "warning", warnings.append)

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert any("not yet supported" in message for message in warnings)


def test_train_segmented_rejects_a_model_missing_required_attributes(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    trainer = StateTrainer(
        model_builder=SimpleNamespace(build=lambda venv, *, state_name: None),
        transition_bonus=0.0,
        config_path=tmp_path / "config.yaml",
    )

    with pytest.raises(TypeError, match="Find is missing"):
        trainer._train_segmented(SimpleNamespace(), {"Find": SimpleNamespace()})


def test_train_segmented_raises_for_an_unregistered_active_state(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.current_state = "Unknown"
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(KeyError, match="Unknown"):
        trainer._train_segmented(venv, {"Find": model})


def test_train_segmented_records_episode_outcome_when_the_model_supports_it(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    outcomes = []
    model.record_episode_outcome = lambda *, fitness, won: outcomes.append((fitness, won))
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert outcomes == [(1.0, True)]


def test_train_segmented_skips_segment_bookkeeping_for_steps_that_do_not_end_a_segment(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.segment_end_after_step = 2
    env.raise_after_steps = 2
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    outcomes = []
    model.record_episode_outcome = lambda *, fitness, won: outcomes.append((fitness, won))
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    # Step 1 does not end a segment and is skipped; only step 2 records an outcome.
    assert outcomes == [(1.0, True)]


def test_train_segmented_restarts_with_fresh_models_when_a_reset_is_requested(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    env = _FakeStateEnv()
    env.raise_after_steps = 1
    venv = DummyVecEnv([lambda: env])
    builder, built = _model_builder()
    model = builder.build(venv, state_name="Find")
    reset_request = object()
    consume_calls = {"count": 0}

    def fake_consume_model_reset():
        consume_calls["count"] += 1
        return reset_request if consume_calls["count"] == 2 else None

    perform_calls = []
    monkeypatch.setattr(state_trainer_module, "consume_model_reset", fake_consume_model_reset)
    monkeypatch.setattr(state_trainer_module, "perform_model_reset", perform_calls.append)
    trainer = StateTrainer(model_builder=builder, transition_bonus=0.0, config_path=tmp_path / "config.yaml")

    with pytest.raises(_StopLoop):
        trainer._train_segmented(venv, {"Find": model})

    assert perform_calls == [reset_request]
    assert len(built) == 2
    assert all(instance.training_end_calls == 1 for instance in _FakeEpisodeCallback.instances[:3])


def test_publish_curriculum_progress_is_a_noop_when_ui_disabled():
    calls = []
    venv = SimpleNamespace(env_method=lambda name: calls.append(name) or [{}])

    StateTrainer._publish_curriculum_progress(venv, SimpleNamespace(ui=SimpleNamespace(enabled=False)))

    assert calls == []


def test_publish_run_savestate_is_a_noop_when_ui_disabled(monkeypatch):
    published = []
    monkeypatch.setattr(
        state_trainer_module,
        "publish_metadata",
        lambda *args, **kwargs: published.append(args),
    )

    StateTrainer._publish_run_savestate(SimpleNamespace(ui=SimpleNamespace(enabled=False)), {}, "Level1")

    assert published == []


def test_update_model_rejects_non_finite_advantages_before_training():
    trainer = object.__new__(StateTrainer)
    trainer.config_path = "unused.yaml"
    model = SimpleNamespace(normalize_advantage=True, train=lambda: pytest.fail("must not train"))
    rollouts = SimpleNamespace(build_buffer=lambda state_name: SimpleNamespace(advantages=np.asarray([[float("nan")]])))

    with pytest.raises(FloatingPointError, match="Find"):
        trainer._update_model("Find", model, rollouts)


def test_savestate_scheduler_rotate_rejects_an_empty_savestate_list():
    scheduler = SavestateScheduler(())

    with pytest.raises(ValueError, match="empty savestate list"):
        scheduler.rotate()


def test_start_episode_callbacks_wires_a_runtime_trainer_and_starts_callbacks(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    trainer = StateTrainer(
        model_builder=SimpleNamespace(build=lambda venv, *, state_name: None),
        transition_bonus=0.0,
        config_path=tmp_path / "config.yaml",
    )
    model = SimpleNamespace()

    callbacks = trainer._start_episode_callbacks({"Find": model}, config, "Level1")

    assert len(callbacks) == 3
    assert _FakeRuntimeTrainer.instances[-1].savestate_calls == ["Level1"]
    assert _FakeRuntimeTrainer.instances[-1].configure_runtime_calls == 1


def test_start_episode_callbacks_skips_savestate_when_none_is_active(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=(), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)
    trainer = StateTrainer(
        model_builder=SimpleNamespace(build=lambda venv, *, state_name: None),
        transition_bonus=0.0,
        config_path=tmp_path / "config.yaml",
    )

    trainer._start_episode_callbacks({"Find": SimpleNamespace()}, config, None)

    assert _FakeRuntimeTrainer.instances[-1].savestate_calls == []


def test_start_segmented_ui_is_a_noop_when_ui_disabled(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=False)
    _patch_collaborators(monkeypatch, config=config)

    StateTrainer._start_segmented_ui(SimpleNamespace(), {}, config, "Level1")


def test_start_segmented_ui_publishes_metadata_when_started(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=True)
    _patch_collaborators(monkeypatch, config=config)
    monkeypatch.setattr(state_trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: None)
    control_calls = []
    monkeypatch.setattr(
        state_trainer_module, "configure_training_control", lambda **kwargs: control_calls.append(kwargs)
    )
    monkeypatch.setattr(state_trainer_module, "start_ui", lambda settings: "server")
    published = []
    monkeypatch.setattr(
        state_trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append(section),
    )
    monkeypatch.setattr(state_trainer_module, "get_model_metadata", lambda model: {"class": "FakeModel"})
    venv = SimpleNamespace(env_method=lambda name: [{}])

    StateTrainer._start_segmented_ui(venv, {"Find": SimpleNamespace()}, config, "Level1")

    assert control_calls[0]["restart_supported"] is True
    assert published == ["run", "models", "savestate_curriculum", "model", "environment"]


def test_start_segmented_ui_skips_publishing_when_the_server_fails_to_start(monkeypatch, tmp_path: Path):
    config = _config(tmp_path, savestates=("Level1",), ui_enabled=True)
    _patch_collaborators(monkeypatch, config=config)
    monkeypatch.setattr(state_trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: None)
    monkeypatch.setattr(state_trainer_module, "configure_training_control", lambda **kwargs: None)
    monkeypatch.setattr(state_trainer_module, "start_ui", lambda settings: None)
    published = []
    monkeypatch.setattr(
        state_trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append(section),
    )

    StateTrainer._start_segmented_ui(SimpleNamespace(), {"Find": SimpleNamespace()}, config, "Level1")

    assert published == []
