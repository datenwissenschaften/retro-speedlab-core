from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from datenwissenschaften import policy_manager as policy_manager_module
from datenwissenschaften.policy_manager import PolicyManager


class FakeModel:
    def __init__(self, name: str):
        self.name = name
        self.calls: list[dict] = []

    def predict(self, observation, **kwargs):
        self.calls.append(kwargs)
        return f"action-{self.name}", f"state-{self.name}"


def test_requires_at_least_one_model():
    with pytest.raises(ValueError, match="at least one model"):
        PolicyManager({})


def test_model_for_raises_for_unknown_state():
    manager = PolicyManager({"Find": FakeModel("Find")})

    with pytest.raises(KeyError, match="Eat"):
        manager.model_for("Eat")


def test_predict_marks_first_call_as_episode_start_and_remembers_recurrent_state():
    model = FakeModel("Find")
    manager = PolicyManager({"Find": model})

    first = manager.predict("obs", state_name="Find")
    manager.predict("obs", state_name="Find")

    assert first == ("action-Find", "state-Find")
    assert model.calls[0]["episode_start"].tolist() == [True]
    assert model.calls[1]["episode_start"].tolist() == [False]
    assert model.calls[1]["state"] == "state-Find"


def test_predict_respects_explicit_kwargs():
    model = FakeModel("Find")
    manager = PolicyManager({"Find": model})

    manager.predict("obs", state_name="Find", state="explicit", episode_start=np.asarray([True]))

    assert model.calls[0]["state"] == "explicit"
    assert model.calls[0]["episode_start"].tolist() == [True]


def test_predict_ignores_non_tuple_predictions():
    class ScalarModel:
        def predict(self, observation, **kwargs):
            return "single-value"

    manager = PolicyManager({"Find": ScalarModel()})

    result = manager.predict("obs", state_name="Find")

    assert result == "single-value"


def test_reset_episode_clears_recurrent_state_and_started_flags():
    model = FakeModel("Find")
    manager = PolicyManager({"Find": model})
    manager.predict("obs", state_name="Find")

    manager.reset_episode()
    manager.predict("obs", state_name="Find")

    assert model.calls[-1]["episode_start"].tolist() == [True]
    assert model.calls[-1]["state"] is None


def test_state_names_lists_registered_models():
    manager = PolicyManager({"Find": FakeModel("Find"), "Eat": FakeModel("Eat")})

    assert manager.state_names() == ["Find", "Eat"]


def test_predict_for_env_uses_callable_state_name():
    model = FakeModel("Find")
    manager = PolicyManager({"Find": model})
    env = SimpleNamespace(state_name=lambda: "Find")

    manager.predict_for_env("obs", env)

    assert model.calls[0]["episode_start"].tolist() == [True]


def test_predict_for_env_uses_env_method_for_vector_envs():
    model = FakeModel("Find")
    manager = PolicyManager({"Find": model})
    env = SimpleNamespace(env_method=lambda name: ["Find"])

    manager.predict_for_env("obs", env)

    assert model.calls[0]["episode_start"].tolist() == [True]


def test_predict_for_env_rejects_multi_environment_vector_envs():
    manager = PolicyManager({"Find": FakeModel("Find")})
    env = SimpleNamespace(env_method=lambda name: ["Find", "Eat"])

    with pytest.raises(ValueError, match="single environment"):
        manager.predict_for_env("obs", env)


def test_predict_for_env_raises_when_state_cannot_be_determined():
    manager = PolicyManager({"Find": FakeModel("Find")})

    with pytest.raises(TypeError, match="state_name"):
        manager.predict_for_env("obs", SimpleNamespace())


def test_load_builds_a_policy_manager_from_configured_state_names(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        policy_manager_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    loaded_paths = []

    def fake_load_model(model_path, env=None):
        loaded_paths.append(model_path)
        return FakeModel(model_path)

    manager = PolicyManager.load(["Find", "Eat"], load_model=fake_load_model)

    assert manager.state_names() == ["Find", "Eat"]
    assert len(loaded_paths) == 2
