import pytest

from datenwissenschaften.core import config as config_module
from datenwissenschaften.core.config import TrainingConfig


def test_valid_configuration_is_accepted():
    config = TrainingConfig(game="Game", num_envs=4, n_stack=4, timestep_batch=1000)

    assert config.num_envs == 4


def test_empty_game_is_rejected():
    with pytest.raises(ValueError, match="game must be set"):
        TrainingConfig(game="", num_envs=4, n_stack=4, timestep_batch=1000)


def test_auto_num_envs_resolves_via_optimal_env_count(monkeypatch):
    monkeypatch.setattr(config_module, "optimal_env_count", lambda: 6)

    config = TrainingConfig(game="Game", num_envs="auto", n_stack=4, timestep_batch=1000)

    assert config.num_envs == 6


def test_non_integer_num_envs_is_rejected():
    with pytest.raises(ValueError, match="num_envs must be at least 1"):
        TrainingConfig(game="Game", num_envs=1.5, n_stack=4, timestep_batch=1000)


def test_boolean_num_envs_is_rejected():
    with pytest.raises(ValueError, match="num_envs must be at least 1"):
        TrainingConfig(game="Game", num_envs=True, n_stack=4, timestep_batch=1000)


def test_zero_num_envs_is_rejected():
    with pytest.raises(ValueError, match="num_envs must be at least 1"):
        TrainingConfig(game="Game", num_envs=0, n_stack=4, timestep_batch=1000)


def test_non_positive_n_stack_is_rejected():
    with pytest.raises(ValueError, match="n_stack must be at least 1"):
        TrainingConfig(game="Game", num_envs=4, n_stack=0, timestep_batch=1000)


def test_non_positive_timestep_batch_is_rejected():
    with pytest.raises(ValueError, match="timestep_batch must be at least 1"):
        TrainingConfig(game="Game", num_envs=4, n_stack=4, timestep_batch=0)
