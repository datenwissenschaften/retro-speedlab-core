import os
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from datenwissenschaften.policy.prior import observation
from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.policy_start import POLICY_START_FILE, load_policy_start
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.search import leaves
from datenwissenschaften.tas.workers import worker_runner

EPISODE_DECISIONS = 128
NOVELTY_REWARD = 0.01
EXIT_REWARD = 10.0
DEATH_PENALTY = 1.0
POLICY_MODE = "policy"
EXIT = "exit"


def reward_of(outcome_reward: float, novelty: int, exited: bool, alive: bool) -> float:
    return outcome_reward + NOVELTY_REWARD * novelty + EXIT_REWARD * exited - DEATH_PENALTY * (not alive)


class PolicyEnv(gymnasium.Env[np.ndarray, np.int64]):
    def __init__(self, wrapper: str, config: Path, output: Path, live: Path) -> None:
        self.runner = worker_runner(wrapper, config)
        self.output = output
        self.live = WorkerLive(live, os.getpid(), POLICY_MODE)
        self.runner.power_on()
        self.observation_space = spaces.Box(0.0, 1.0, (len(self.runner.memory()),), np.float32)
        self.action_space = spaces.Discrete(len(self.runner.wrapper.action_table))
        self.start: Start | None = None
        self.novelty = RamNovelty(self.runner.memory())
        self.pressed: list[np.ndarray] = []
        self.episodes = 0

    def begin(self) -> Start:
        start = load_policy_start(self.output / POLICY_START_FILE)
        if start is None:
            raise RuntimeError("The policy trains only from a start the search has saved")
        self.runner.restore(start)
        if start != self.start:
            self.start, self.novelty = start, RamNovelty(self.runner.memory())
        return start

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.begin()
        self.pressed = []
        return observation(self.runner.memory()), {}

    def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        start = self.start if self.start is not None else self.begin()
        outcome = self.runner.decide(int(action))
        self.pressed.append(outcome.pressed)
        self.live.step(self.runner, self.episodes, None, len(outcome.pressed))
        exited = leaves(self.runner, start, outcome.state, outcome.won)
        reward = reward_of(outcome.reward, self.novelty.see(self.runner.memory()), exited, outcome.alive)
        terminated = exited or not outcome.alive or outcome.state != start.state
        truncated = len(self.pressed) >= EPISODE_DECISIONS
        self.episodes += terminated or truncated
        info = {EXIT: (start.state, np.concatenate(self.pressed))} if exited else {}
        return observation(self.runner.memory()), reward, terminated, truncated, info
