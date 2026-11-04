import io
import multiprocessing
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import wandb
from loguru import logger
from numpy.typing import NDArray
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from torch import nn

from datenwissenschaften.experiments import start_experiment
from datenwissenschaften.gpu import gpu
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.policy.env import EXIT, PolicyEnv
from datenwissenschaften.policy.prior import BIAS, POLICY_FILE, WEIGHT, Weights
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.live import LIVE_DIR
from datenwissenschaften.tas.paths import atomic_write, tas_dir
from datenwissenschaften.tas.policy_start import POLICY_EXITS_FILE, POLICY_START_FILE, save_policy_exit

ROLLOUT_DECISIONS = 1024
HIDDEN = 256
MINIBATCH = 256
EPOCHS = 4
LEARNING_RATE = 3e-4
ENTROPY_WEIGHT = 0.01
WAIT_SECONDS = 30.0
POLICY_LOG = "policy.log"
POLICY_JOB = "policy"
SPAWN = "spawn"


def linear(layer: nn.Module) -> nn.Linear:
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"The policy head is a {type(layer).__name__}, not a linear layer")
    return layer


def policy_weights(policy: ActorCriticPolicy) -> Weights:
    hidden = [layer for layer in policy.mlp_extractor.policy_net if isinstance(layer, nn.Linear)]
    weights: Weights = {}
    for index, layer in enumerate([*hidden, linear(policy.action_net)]):
        weights[f"{WEIGHT}{index}"] = layer.weight.detach().cpu().numpy()
        weights[f"{BIAS}{index}"] = layer.bias.detach().cpu().numpy()
    return weights


def export_policy(policy: ActorCriticPolicy, path: Path) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, allow_pickle=True, **policy_weights(policy))
    atomic_write(path, buffer.getvalue())


def policy_model(env: VecEnv, device: torch.device) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        n_steps=ROLLOUT_DECISIONS,
        batch_size=MINIBATCH,
        n_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        ent_coef=ENTROPY_WEIGHT,
        policy_kwargs={"net_arch": {"pi": [HIDDEN, HIDDEN], "vf": [HIDDEN, HIDDEN]}, "activation_fn": nn.Tanh},
        device=device,
    )


class RoundCallback(BaseCallback):
    def __init__(self, output: Path, experiment: wandb.Run) -> None:
        super().__init__()
        self.output = output
        self.experiment = experiment
        self.rounds = 0
        self.rewards = 0.0
        self.episodes = 0
        self.exits = 0

    def _on_step(self) -> bool:
        rewards: NDArray[np.float32] = self.locals["rewards"]
        dones: NDArray[np.bool_] = self.locals["dones"]
        self.rewards += float(np.sum(rewards))
        self.episodes += int(np.sum(dones))
        for info in self.locals["infos"]:
            if EXIT in info:
                self.exits += 1
                self.keep_exit(*info[EXIT])
        return True

    def keep_exit(self, state: str, pressed: np.ndarray) -> None:
        if save_policy_exit(self.output / POLICY_EXITS_FILE, state, pressed):
            logger.info(f"The policy found a way out of {state}; the search replays it next")

    def _on_rollout_start(self) -> None:
        if self.episodes == 0:
            return
        policy = self.model.policy
        if not isinstance(policy, ActorCriticPolicy):
            raise TypeError(f"The policy is a {type(policy).__name__}, not an actor-critic policy")
        export_policy(policy, self.output / POLICY_FILE)
        self.rounds += 1
        reward = self.rewards / self.episodes
        trained = {f"policy/{name.split('/')[-1]}": value for name, value in self.logger.name_to_value.items()}
        self.experiment.log(
            {"policy/reward": reward, "policy/exits": self.exits, "policy/episodes": self.episodes, **trained}
        )
        logger.info(
            f"Policy round {self.rounds}: reward {reward:.2f} per episode, {self.exits} exits in "
            f"{self.episodes} episodes"
        )
        self.rewards, self.episodes, self.exits = 0.0, 0, 0


def speedrun_alive() -> bool:
    speedrun = multiprocessing.parent_process()
    if speedrun is None:
        raise RuntimeError("The policy trains only in a process the speedrun started")
    return speedrun.is_alive()


def wait_for_start(output: Path) -> bool:
    while not (output / POLICY_START_FILE).is_file():
        if not speedrun_alive():
            return False
        time.sleep(WAIT_SECONDS)
    return True


def train_policy_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    output = tas_dir(settings)
    logger.add(output / POLICY_LOG)
    if not wait_for_start(output):
        return
    make = partial(PolicyEnv, wrapper, config, output, output / LIVE_DIR)
    env = SubprocVecEnv([make] * settings.speedrun.policy_emulators, start_method=SPAWN)
    model = policy_model(env, gpu())
    callback = RoundCallback(output, start_experiment(settings, POLICY_JOB))
    while speedrun_alive():
        model.learn(ROLLOUT_DECISIONS * env.num_envs, callback=callback, reset_num_timesteps=False)  # pyright: ignore[reportUnknownMemberType]
    env.close()
    logger.warning("The speedrun has stopped, so the policy stops with it")
