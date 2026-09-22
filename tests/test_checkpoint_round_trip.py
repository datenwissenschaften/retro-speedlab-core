import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from datenwissenschaften.rnd.model import AdaptiveRecurrentRNDPPO

_VISUAL_SHAPE = (3, 40, 40)


class _FakeGameEnv(gym.Env):
    """A minimal Dict-observation environment; no emulator or ROM required."""

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
        terminated = self._steps >= 6
        return self.observation_space.sample(), 0.1, terminated, False, {"won": False}


def _build_model() -> AdaptiveRecurrentRNDPPO:
    venv = DummyVecEnv([_FakeGameEnv])
    return AdaptiveRecurrentRNDPPO(
        "MultiInputLstmPolicy",
        venv,
        n_steps=6,
        batch_size=6,
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


def test_cpu_model_constructs_and_trains_one_rollout():
    model = _build_model()

    model.learn(total_timesteps=6)

    assert model.num_timesteps >= 6
    assert model.device.type == "cpu"


def test_checkpoint_round_trip_restores_policy_and_rnd_state():
    model = _build_model()
    model.learn(total_timesteps=6)
    # Simulate exploration having adapted away from the 1x baseline before saving.
    model.adaptation_multiplier = 2.5
    model._attach_rnd_to_env()
    observations_seen_before = model.rnd.observations_seen.item()

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = str(Path(tmp_dir) / "model")
        model.save(checkpoint_path)
        restored = AdaptiveRecurrentRNDPPO.load(checkpoint_path, env=model.get_env(), device="cpu")

    # Adaptation state (annealing progress, adaptation multiplier) survives the round trip.
    assert restored.adaptation_multiplier == model.adaptation_multiplier
    assert restored.rnd.adaptation_multiplier == model.rnd.adaptation_multiplier
    assert restored.rnd.observations_seen.item() == observations_seen_before

    # Reward normalization statistics survive the round trip.
    assert torch.equal(restored.rnd.reward_mean, model.rnd.reward_mean)
    assert torch.equal(restored.rnd.reward_variance, model.rnd.reward_variance)
    assert torch.equal(restored.rnd.reward_count, model.rnd.reward_count)

    # Both RND networks and the policy are restored exactly.
    assert all(torch.equal(a, b) for a, b in zip(model.rnd.predictor.parameters(), restored.rnd.predictor.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(model.rnd.target.parameters(), restored.rnd.target.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(model.policy.parameters(), restored.policy.parameters()))

    # The restored target network is still frozen.
    assert all(not parameter.requires_grad for parameter in restored.rnd.target.parameters())


def test_checkpoint_survives_a_second_training_chunk_after_restore():
    model = _build_model()
    model.learn(total_timesteps=6)

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = str(Path(tmp_dir) / "model")
        model.save(checkpoint_path)
        restored = AdaptiveRecurrentRNDPPO.load(checkpoint_path, env=model.get_env(), device="cpu")

    restored.learn(total_timesteps=6, reset_num_timesteps=False)

    assert restored.num_timesteps >= model.num_timesteps + 6
