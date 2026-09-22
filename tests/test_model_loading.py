from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from datenwissenschaften import model as model_module
from datenwissenschaften.model import (
    ModelBuilder,
    get_model_metadata,
    get_model_path,
    load_or_create_model,
    model_parameters_are_finite,
)


def test_datenwissenschaften_version_falls_back_when_package_is_not_installed(monkeypatch):
    def fake_version(name):
        raise model_module.PackageNotFoundError(name)

    monkeypatch.setattr(model_module, "version", fake_version)

    assert model_module.datenwissenschaften_version() == "DEVELOPMENT"


def test_model_parameters_are_finite_returns_true_without_get_parameters():
    assert model_parameters_are_finite(object()) is True


def test_model_parameters_are_finite_detects_nan_in_nested_structures():
    model = SimpleNamespace(get_parameters=lambda: {"policy": {"weights": [torch.tensor([1.0, float("nan")])]}})

    assert model_parameters_are_finite(model) is False


def test_model_parameters_are_finite_ignores_non_floating_point_tensors():
    model = SimpleNamespace(get_parameters=lambda: {"counter": torch.tensor([1, 2, 3])})

    assert model_parameters_are_finite(model) is True


def test_get_model_path_creates_the_parent_directory(tmp_path: Path):
    path = get_model_path(str(tmp_path), "Game", "Level1")

    assert path == str(tmp_path / "Game" / "Level1" / "model")
    assert (tmp_path / "Game" / "Level1").is_dir()


def test_get_model_metadata_includes_ppo_and_rnd_fields_when_present():
    rnd = SimpleNamespace(observations_seen=torch.tensor(42), coefficient=0.5)
    model = SimpleNamespace(
        action_space="Discrete(4)",
        device="cpu",
        n_envs=2,
        num_timesteps=100,
        observation_space="Box(...)",
        display_name="Test Model",
        description="A test model",
        batch_size=64,
        n_steps=128,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        learning_rate=0.0003,
        policy_kwargs={},
        policy=SimpleNamespace(),
        rnd_output_size=8,
        rnd_learning_rate=1e-4,
        rnd_update_proportion=0.25,
        rnd_gamma=0.99,
        rnd_intrinsic_coefficient=0.5,
        rnd_final_intrinsic_coefficient=0.02,
        rnd_anneal_steps=1_000,
        rnd_reward_clip=1.0,
        rnd=rnd,
        adaptation_multiplier=1.5,
        adaptation_reason="progressing",
        adaptive_autoconfigure=True,
        adaptive_action_count=4,
        adaptive_observation_pixels=100,
        adaptive_rollout_steps=128,
        adaptive_score_delta=0.1,
        adaptive_score_staleness_episodes=25,
        adaptive_no_win_staleness_episodes=50,
        adaptive_multiplier_min=0.5,
        adaptive_multiplier_max=4.0,
        adaptive_learning_rate_min=1e-5,
        adaptive_learning_rate_max=1e-3,
        adaptive_rnd_update_proportion=0.3,
        adaptive_rnd_update_max=1.0,
        episodes_since_score_improvement=3,
        episodes_since_win=1,
    )

    metadata = get_model_metadata(model)

    assert metadata["n_envs"] == 2
    assert metadata["ppo"]["batch_size"] == 64
    assert metadata["rnd"]["observations_seen"] == 42
    assert metadata["rnd"]["current_intrinsic_coefficient"] == 0.5


def test_get_model_metadata_omits_ppo_and_rnd_sections_when_fields_are_absent():
    metadata = get_model_metadata(SimpleNamespace())

    assert "ppo" not in metadata
    assert "rnd" not in metadata


def test_get_model_metadata_rnd_defaults_observations_seen_when_rnd_is_none():
    model = SimpleNamespace(rnd_output_size=8, rnd=None)

    metadata = get_model_metadata(model)

    assert metadata["rnd"]["observations_seen"] == 0
    assert metadata["rnd"]["current_intrinsic_coefficient"] is None


def test_load_or_create_model_builds_fresh_when_no_checkpoint_exists(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    monkeypatch.setattr(
        model_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)
    built = []

    def build_model(venv):
        built.append(venv)
        return "fresh-model"

    result = load_or_create_model(
        "venv",
        build_model=build_model,
        load_model=lambda *args, **kwargs: pytest.fail("must not load"),
        state_name="Level1",
    )

    assert result == "fresh-model"
    assert built == ["venv"]


def test_load_or_create_model_loads_existing_checkpoint(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    monkeypatch.setattr(
        model_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)
    model_dir = tmp_path / "Game" / "Level1"
    model_dir.mkdir(parents=True)
    (model_dir / "model.zip").write_bytes(b"fake-zip")
    loaded_model = SimpleNamespace(get_parameters=lambda: {})

    def load_model(model_path, *, env, verbose):
        assert model_path.endswith("model")
        return loaded_model

    result = load_or_create_model(
        "venv",
        build_model=lambda venv: pytest.fail("must not build"),
        load_model=load_model,
        state_name="Level1",
    )

    assert result is loaded_model


def test_load_or_create_model_discards_a_checkpoint_that_fails_to_load(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    monkeypatch.setattr(
        model_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)
    model_dir = tmp_path / "Game" / "Level1"
    model_dir.mkdir(parents=True)
    (model_dir / "model.zip").write_bytes(b"corrupt")

    def load_model(model_path, *, env, verbose):
        raise RuntimeError("corrupt checkpoint")

    result = load_or_create_model(
        "venv",
        build_model=lambda venv: "fresh-model",
        load_model=load_model,
        state_name="Level1",
    )

    assert result == "fresh-model"
    assert not (model_dir / "model.zip").exists()


def test_load_or_create_model_discards_a_checkpoint_with_non_finite_parameters(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    monkeypatch.setattr(
        model_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)
    model_dir = tmp_path / "Game" / "Level1"
    model_dir.mkdir(parents=True)
    (model_dir / "model.zip").write_bytes(b"fake-zip")
    broken_model = SimpleNamespace(get_parameters=lambda: {"weight": torch.tensor([float("inf")])})

    result = load_or_create_model(
        "venv",
        build_model=lambda venv: "fresh-model",
        load_model=lambda model_path, *, env, verbose: broken_model,
        state_name="Level1",
    )

    assert result == "fresh-model"


def test_load_or_create_model_runs_cleanup_hook_when_available(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    config = SimpleNamespace(
        paths=SimpleNamespace(models_dir=tmp_path),
        training=SimpleNamespace(game_identity="Game"),
    )
    monkeypatch.setattr(model_module, "load_config", lambda path: config)
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)
    cleanup_calls = []

    def build_model(venv):
        return "fresh-model"

    build_model.cleanup_incompatible_artifacts = cleanup_calls.append

    result = load_or_create_model(
        "venv",
        build_model=build_model,
        load_model=lambda *args, **kwargs: pytest.fail("must not load"),
        state_name="Level1",
    )

    assert result == "fresh-model"
    assert cleanup_calls == [config]


def test_model_builder_instantiates_a_type_and_uses_its_load_method(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(model_module, "configure_accelerator", lambda: None)
    monkeypatch.setattr(
        model_module,
        "load_config",
        lambda path: SimpleNamespace(
            paths=SimpleNamespace(models_dir=tmp_path),
            training=SimpleNamespace(game_identity="Game"),
        ),
    )
    monkeypatch.setattr(model_module, "reset_for_training_change", lambda config, venv, config_path: None)

    class FactoryModel:
        def __call__(self, venv):
            return "fresh-model"

        @staticmethod
        def load(model_path, *, env, verbose):
            pytest.fail("must not load")

    builder = ModelBuilder(FactoryModel)

    result = builder.build("venv", state_name="Level1")

    assert result == "fresh-model"


def test_class_path_prefers_module_qualname_for_plain_functions():
    def build_model(venv):
        return None

    assert model_module._class_path(build_model) == f"{build_model.__module__}.{build_model.__qualname__}"


def test_class_path_falls_back_to_the_instances_class():
    assert model_module._class_path(np.zeros(1)) == "numpy.ndarray"


def test_class_path_uses_the_class_itself_when_given_a_type():
    assert model_module._class_path(ModelBuilder) == f"{ModelBuilder.__module__}.{ModelBuilder.__qualname__}"


def test_class_path_falls_back_to_the_class_for_callables_without_a_qualname():
    class CallableWithoutQualname:
        def __call__(self, venv):
            return None

    instance = CallableWithoutQualname()

    assert model_module._class_path(instance) == f"{instance.__module__}.{instance.__class__.__qualname__}"
