import gymnasium as gym
import numpy as np
import pytest
import torch

from datenwissenschaften.rnd.model import _RandomNetworkDistillation

_VISUAL_SHAPE = (3, 40, 40)


def _observation_space() -> gym.spaces.Dict:
    return gym.spaces.Dict(
        {
            "visual": gym.spaces.Box(low=0, high=255, shape=_VISUAL_SHAPE, dtype=np.uint8),
            "ram": gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
        }
    )


def _rnd(**overrides) -> _RandomNetworkDistillation:
    kwargs = {
        "output_size": 8,
        "learning_rate": 1e-3,
        "update_proportion": 1.0,
        "intrinsic_gamma": 0.99,
        "intrinsic_coefficient": 1.0,
        "final_intrinsic_coefficient": 1.0,
        "anneal_steps": 1000,
        "reward_clip": 5.0,
        "device": torch.device("cpu"),
        **overrides,
    }
    return _RandomNetworkDistillation(_observation_space(), **kwargs)


def _observations(num_envs: int) -> dict[str, np.ndarray]:
    return {"visual": np.random.randint(0, 255, size=(num_envs, *_VISUAL_SHAPE), dtype=np.uint8)}


def test_target_network_never_trains_while_predictor_does():
    rnd = _rnd()
    target_before = [parameter.clone() for parameter in rnd.target.parameters()]
    predictor_before = [parameter.clone() for parameter in rnd.predictor.parameters()]

    observations = _observations(4)
    dones = np.array([False, False, False, True])
    for _ in range(5):
        rnd.intrinsic_reward(observations, dones)

    assert all(torch.equal(before, after) for before, after in zip(target_before, rnd.target.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(predictor_before, rnd.predictor.parameters()))


def test_target_network_parameters_are_frozen():
    rnd = _rnd()
    assert all(not parameter.requires_grad for parameter in rnd.target.parameters())
    assert rnd.target.training is False


def test_checkpoint_round_trip_restores_weights_and_reward_statistics():
    rnd = _rnd()
    observations = _observations(4)
    dones = np.array([False, False, False, True])
    for _ in range(5):
        rnd.intrinsic_reward(observations, dones)

    restored = _rnd()
    restored.load_state_dict(rnd.state_dict())

    assert all(torch.equal(a, b) for a, b in zip(rnd.predictor.parameters(), restored.predictor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(rnd.target.parameters(), restored.target.parameters()))
    assert torch.equal(rnd.reward_mean, restored.reward_mean)
    assert torch.equal(rnd.reward_variance, restored.reward_variance)
    assert torch.equal(rnd.reward_count, restored.reward_count)
    assert restored.observations_seen.item() == rnd.observations_seen.item()


def test_checkpoint_round_trip_does_not_restore_adaptation_multiplier():
    # ``adaptation_multiplier`` is plain Python state, not part of the torch
    # state_dict; ``AdaptiveRecurrentRNDPPO._attach_rnd_to_env`` is responsible
    # for reapplying it after a checkpoint restore. See the comment there.
    rnd = _rnd()
    rnd.set_adaptation_multiplier(2.5)

    restored = _rnd()
    restored.load_state_dict(rnd.state_dict())

    assert restored.adaptation_multiplier == 1.0


@pytest.mark.parametrize(
    "field, value",
    [
        ("update_proportion", 0.0),
        ("update_proportion", 1.5),
        ("intrinsic_gamma", -0.1),
        ("intrinsic_gamma", 1.1),
        ("intrinsic_coefficient", -1.0),
        ("final_intrinsic_coefficient", -1.0),
        ("anneal_steps", 0),
        ("reward_clip", 0.0),
        ("reward_clip", -1.0),
    ],
)
def test_invalid_rnd_hyperparameters_are_rejected(field, value):
    with pytest.raises(ValueError):
        _rnd(**{field: value})


def test_rnd_requires_a_visual_dict_observation():
    space = gym.spaces.Dict({"ram": gym.spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)})
    with pytest.raises(ValueError, match="visual"):
        _RandomNetworkDistillation(
            space,
            output_size=8,
            learning_rate=1e-3,
            update_proportion=1.0,
            intrinsic_gamma=0.99,
            intrinsic_coefficient=1.0,
            final_intrinsic_coefficient=1.0,
            anneal_steps=1000,
            reward_clip=5.0,
            device=torch.device("cpu"),
        )
