import os
import random
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from datenwissenschaften.policy.prior import observation
from datenwissenschaften.policy.starts import Opening, chosen, openings
from datenwissenschaften.speedrun.live import PPO, WorkerLive
from datenwissenschaften.speedrun.novelty import RamNovelty
from datenwissenschaften.speedrun.paths import ARCHIVES_DIR
from datenwissenschaften.speedrun.policy_start import POLICY_START_FILE, load_policy_start
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import leaves
from datenwissenschaften.speedrun.workers import worker_runner

EPISODE_DECISIONS = 512
RELOAD_EPISODES = 64
NOVELTY_REWARD = 0.01
EXIT_REWARD = 100.0
DEATH_PENALTY = 1.0
POLICY_MODE = "policy"
EXIT = "exit"


def reward_of(outcome_reward: float, novelty: int, exited: bool, alive: bool) -> float:
    return outcome_reward + NOVELTY_REWARD * novelty + EXIT_REWARD * exited - DEATH_PENALTY * (not alive)


class PolicyEnv(gymnasium.Env[np.ndarray, np.int64]):
    def __init__(self, wrapper: str, config: Path, output: Path, live: Path) -> None:
        self.runner = worker_runner(wrapper, config)
        self.output = output
        self.live = WorkerLive(live, os.getpid(), PPO, POLICY_MODE)
        self.runner.power_on()
        self.observation_space = spaces.Box(0.0, 1.0, (len(self.runner.memory()),), np.float32)
        self.action_space = spaces.Discrete(len(self.runner.wrapper.action_table))
        self.buttons = self.runner.wrapper.action_table.shape[-1]
        self.rng = random.Random(os.getpid())
        self.start: Start | None = None
        self.novelty = RamNovelty(self.runner.memory())
        self.openings: list[Opening] = []
        self.loaded_at = 0
        self.pressed: list[np.ndarray] = []
        self.decisions = 0
        self.episodes = 0

    def begin(self) -> Start:
        start = load_policy_start(self.output / POLICY_START_FILE)
        if start is None:
            raise RuntimeError("The policy trains only from a start the search has saved")
        if start != self.start:
            self.runner.restore(start)
            self.start, self.novelty = start, RamNovelty(self.runner.memory())
            self.openings = []
        if not self.openings or self.episodes >= self.loaded_at + RELOAD_EPISODES:
            self.openings, self.loaded_at = openings(self.output / ARCHIVES_DIR, start, self.buttons), self.episodes
        opening = chosen(self.openings, self.rng)
        self.runner.restore(opening.start)
        self.pressed, self.decisions = [opening.prefix], 0
        return start

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.begin()
        return observation(self.runner.memory()), {}

    def step(self, action: np.int64) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        start = self.start if self.start is not None else self.begin()
        outcome = self.runner.decide(int(action))
        self.pressed.append(outcome.pressed)
        self.decisions += 1
        self.live.step(self.runner, self.episodes, None, len(outcome.pressed), None)
        exited = leaves(self.runner, start, outcome.state, outcome.won)
        reward = reward_of(outcome.reward, self.novelty.see(self.runner.memory()), exited, outcome.alive)
        terminated = exited or not outcome.alive or outcome.state != start.state
        truncated = self.decisions >= EPISODE_DECISIONS
        self.episodes += terminated or truncated
        info = {EXIT: (start.state, np.concatenate(self.pressed))} if exited else {}
        return observation(self.runner.memory()), reward, terminated, truncated, info
