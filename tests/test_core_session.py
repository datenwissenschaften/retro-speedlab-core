from types import SimpleNamespace

import pytest

from datenwissenschaften.core import session as session_module
from datenwissenschaften.core.session import TrainingSession


class _StopLoop(Exception):
    pass


class FakeModel:
    num_timesteps = 0

    def __init__(self):
        self.learn_calls: list[dict] = []

    def learn(self, *, total_timesteps, callback, reset_num_timesteps):
        self.learn_calls.append(
            {
                "total_timesteps": total_timesteps,
                "callback": callback,
                "reset_num_timesteps": reset_num_timesteps,
            }
        )
        raise _StopLoop


def _config(*, num_envs=2, n_stack=4, timestep_batch=1_000):
    return SimpleNamespace(game="Game", num_envs=num_envs, n_stack=n_stack, timestep_batch=timestep_batch)


def test_build_configures_accelerator_and_builds_the_venv_and_model(monkeypatch):
    accelerator_calls = []
    monkeypatch.setattr(session_module, "configure_accelerator", lambda: accelerator_calls.append(True))
    built_venv_args = []
    model = FakeModel()

    def build_venv(num_envs, n_stack):
        built_venv_args.append((num_envs, n_stack))
        return "venv"

    def build_model(venv):
        assert venv == "venv"
        return model

    session = TrainingSession(config=_config(), build_venv=build_venv, build_model=build_model)

    assert session.build() is model
    assert built_venv_args == [(2, 4)]
    assert accelerator_calls == [True]


def test_initialize_claims_the_game_and_runs_initializers():
    initialized = []
    session = TrainingSession(
        config=_config(),
        build_venv=lambda num_envs, n_stack: None,
        build_model=lambda venv: None,
        initializers=[initialized.append],
    )

    session.initialize()

    assert initialized == ["Game"]
    assert session._active_game == "Game"


def test_claiming_a_second_game_raises():
    session = TrainingSession(
        config=_config(),
        build_venv=lambda num_envs, n_stack: None,
        build_model=lambda venv: None,
    )
    session.initialize()

    with pytest.raises(RuntimeError, match="already bound"):
        session._claim_game("OtherGame")


def test_train_forever_calls_model_learn_with_the_configured_callbacks():
    model = FakeModel()
    callback_factory_calls = []

    def callbacks():
        callback_factory_calls.append(True)
        return ["callback-a"]

    session = TrainingSession(
        config=_config(timestep_batch=500),
        build_venv=lambda num_envs, n_stack: None,
        build_model=lambda venv: None,
        callbacks=callbacks,
    )

    with pytest.raises(_StopLoop):
        session.train_forever(model)

    assert model.learn_calls == [{"total_timesteps": 500, "callback": ["callback-a"], "reset_num_timesteps": False}]
    assert callback_factory_calls == [True]


def test_run_initializes_builds_and_trains():
    model = FakeModel()
    session = TrainingSession(
        config=_config(),
        build_venv=lambda num_envs, n_stack: "venv",
        build_model=lambda venv: model,
    )

    with pytest.raises(_StopLoop):
        session.run()

    assert session._active_game == "Game"
    assert model.learn_calls
