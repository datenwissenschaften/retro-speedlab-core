from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften import trainer as trainer_module
from datenwissenschaften.trainer import Trainer


class FakeRedisStore:
    def __init__(self, redis_url):
        self.redis_url = redis_url
        self.deleted_prefixes: list[tuple] = []
        self.deleted: list[tuple] = []
        self.values: dict[tuple, object] = {}

    def delete_prefix(self, *parts):
        self.deleted_prefixes.append(parts)

    def delete(self, *parts):
        self.deleted.append(parts)

    def get(self, *parts, default=None):
        return self.values.get(parts, default)

    def set(self, *parts, value):
        self.values[parts] = value


class _StopLoop(Exception):
    pass


class FakeModel:
    def __init__(self):
        self.learn_calls: list[dict] = []
        self.supports_ui_restart = True

    def learn(self, *, total_timesteps, callback, reset_num_timesteps):
        self.learn_calls.append(
            {
                "total_timesteps": total_timesteps,
                "callback": callback,
                "reset_num_timesteps": reset_num_timesteps,
            }
        )
        raise _StopLoop

    def get_env(self):
        return None


def _build_trainer(monkeypatch, tmp_path: Path, *, ui_enabled: bool = False) -> Trainer:
    config = SimpleNamespace(
        paths=SimpleNamespace(
            models_dir=tmp_path / "models", record_dir=tmp_path / "recordings", cache_dir=tmp_path / "cache"
        ),
        training=SimpleNamespace(
            game="Game",
            game_identity="Game",
            active_savestate="Level1",
            savestates=("Level1", "Level2"),
            num_envs=2,
        ),
        log_level="INFO",
        upload=SimpleNamespace(url="https://example.test", api_key=None),
        ui=SimpleNamespace(
            enabled=ui_enabled,
            redis_url="redis://example",
            history_key_prefix="datenwissenschaften:history",
            max_episodes=100,
        ),
    )
    monkeypatch.setattr(trainer_module, "load_config", lambda path: config)
    monkeypatch.setattr(trainer_module, "setup_logging", lambda level: None)
    monkeypatch.setattr(trainer_module, "RedisStore", FakeRedisStore)
    monkeypatch.setattr(trainer_module.Trainer, "_default_callbacks", lambda self: [])
    return Trainer(config_path=tmp_path / "config.yaml", state_name="Level1")


def test_init_builds_default_and_additional_callbacks(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        trainer_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path, record_dir=tmp_path, cache_dir=tmp_path),
            training=SimpleNamespace(game="Game", game_identity="Game", active_savestate="Level1"),
            log_level="INFO",
            upload=SimpleNamespace(url="https://example.test", api_key=None),
            ui=SimpleNamespace(enabled=False, redis_url="redis://example", history_key_prefix="prefix", max_episodes=1),
        ),
    )
    monkeypatch.setattr(trainer_module, "setup_logging", lambda level: None)
    monkeypatch.setattr(trainer_module, "RedisStore", FakeRedisStore)

    trainer = Trainer(config_path=tmp_path / "config.yaml", state_name="Level1", additional_callbacks=["extra"])

    assert trainer.callbacks[-1] == "extra"
    assert trainer._savestate == "Level1"


def test_train_loops_until_model_learn_raises(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path)
    accelerator_calls = []
    monkeypatch.setattr(trainer_module, "configure_accelerator", lambda: accelerator_calls.append(True))
    monkeypatch.setattr(trainer, "_start_ui", lambda model: None)
    model = FakeModel()

    with pytest.raises(_StopLoop):
        trainer.train(model)

    assert accelerator_calls == [True]
    assert model.learn_calls == [
        {"total_timesteps": trainer.training_chunk_steps, "callback": [], "reset_num_timesteps": False}
    ]


def test_start_ui_is_a_noop_when_ui_disabled(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path, ui_enabled=False)

    trainer._start_ui(FakeModel())


def test_start_ui_publishes_metadata_when_started(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path, ui_enabled=True)
    history_calls = []
    monkeypatch.setattr(
        trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: history_calls.append(scope)
    )
    control_calls = []
    monkeypatch.setattr(
        trainer_module,
        "configure_training_control",
        lambda **kwargs: control_calls.append(kwargs),
    )
    monkeypatch.setattr(trainer_module, "start_ui", lambda settings: "server")
    published = []
    monkeypatch.setattr(
        trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append((section, values, replace)),
    )
    monkeypatch.setattr(trainer_module, "get_model_metadata", lambda model: {"class": "FakeModel"})

    trainer._start_ui(FakeModel())

    assert history_calls == ["Game"]
    assert control_calls[0]["game"] == "Game"
    assert [section for section, _, _ in published] == ["run", "model", "environment"]


def test_start_ui_skips_publishing_when_server_fails_to_start(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path, ui_enabled=True)
    monkeypatch.setattr(trainer_module, "configure_history", lambda scope, *, redis_url, key_prefix: None)
    monkeypatch.setattr(trainer_module, "configure_training_control", lambda **kwargs: None)
    monkeypatch.setattr(trainer_module, "start_ui", lambda settings: None)
    published = []
    monkeypatch.setattr(
        trainer_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append(section),
    )

    trainer._start_ui(FakeModel())

    assert published == []


def test_reset_for_restart_resets_savestate_callbacks_and_environment(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path)
    trainer._savestate = "Level2"
    env_calls = []
    model = SimpleNamespace(get_env=lambda: SimpleNamespace(env_method=lambda name: env_calls.append(name)))

    trainer._reset_for_restart(model)

    assert trainer._savestate == "Level1"
    assert env_calls == ["reset_training_memory"]
    assert trainer._store.deleted_prefixes == [("state", "Game"), ("target-memory", "Game")]


def test_reset_for_restart_tolerates_an_environment_without_env_method(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path)
    model = SimpleNamespace(get_env=lambda: None)

    trainer._reset_for_restart(model)


def test_environment_metadata_handles_a_missing_environment():
    assert Trainer._environment_metadata(None) == {"class": None}


def test_environment_metadata_walks_the_wrapper_chain_and_reports_discrete_actions():
    class Inner:
        action_space = "Box(1,)"

        def env_method(self, name):
            return [4]

    class Outer:
        venv = Inner()
        observation_space = "Dict(...)"
        action_space = "Box(1,)"
        num_envs = 2

        def env_method(self, name):
            return self.venv.env_method(name)

    metadata = Trainer._environment_metadata(Outer())

    assert metadata["num_envs"] == 2
    assert metadata["action_space"] == "Discrete(4)"
    assert metadata["emulator_action_space"] == "Box(1,)"
    assert len(metadata["wrappers"]) == 2


def test_environment_metadata_falls_back_to_the_emulator_action_space_when_num_actions_is_invalid():
    class Env:
        action_space = "Box(1,)"

        def env_method(self, name):
            return ["not-a-count"]

    metadata = Trainer._environment_metadata(Env())

    assert metadata["action_space"] == "Box(1,)"


def test_environment_metadata_stops_walking_when_env_points_to_itself():
    class SelfReferential:
        action_space = "Discrete(2)"

        def env_method(self, name):
            raise AttributeError

    env = SelfReferential()
    env.env = env

    metadata = Trainer._environment_metadata(env)

    assert metadata["wrappers"] == [f"{env.__class__.__module__}.{env.__class__.__qualname__}"]


def test_configure_runtime_registers_state_accessors(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path)
    monkeypatch.setattr(trainer_module, "get_last_environment_wrapper", lambda: "wrapper-class")
    configured = []
    monkeypatch.setattr(trainer_module, "configure_runtime", configured.append)

    trainer._configure_runtime()

    runtime = configured[0]
    assert runtime.wrappers == {"Game": "wrapper-class"}
    assert runtime.get_game() == "Game"
    assert runtime.get_savestate() == "Level1"
    runtime.set_savestate("Level2")
    assert trainer._savestate == "Level2"
    trainer._set_state_value("won", True)
    assert trainer._get_state_value("won") is True


def test_get_state_value_defaults_to_false_when_unset(monkeypatch, tmp_path: Path):
    trainer = _build_trainer(monkeypatch, tmp_path)

    assert trainer._get_state_value("missing") is False
