import multiprocessing
import os
import time
from concurrent.futures import Executor
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.gpu import gpu
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.policy.collector import collect_rollout
from datenwissenschaften.policy.network import PolicyNet, export_policy, numpy_weights
from datenwissenschaften.policy.ppo import update
from datenwissenschaften.policy.prior import POLICY_FILE
from datenwissenschaften.policy.rollout import ROLLOUT_DECISIONS, Rollout, merged
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.live import LIVE_DIR
from datenwissenschaften.tas.paths import tas_dir
from datenwissenschaften.tas.policy_start import (
    POLICY_EXITS_FILE,
    POLICY_START_FILE,
    load_policy_start,
    save_policy_exit,
)
from datenwissenschaften.tas.run import worker_pool
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.workflow.wrapper import wrapper_class

LEARNING_RATE = 3e-4
WAIT_SECONDS = 30.0
POLICY_LOG = "policy.log"


@dataclass(slots=True, frozen=True)
class Collectors:
    wrapper: str
    config: Path
    pool: Executor
    emulators: int
    live: Path


class PolicyTrainer:
    def __init__(self, runner: SegmentRunner, output: Path, device: torch.device, collectors: Collectors) -> None:
        self.output = output
        self.device = device
        self.collectors = collectors
        runner.power_on()
        self.net = PolicyNet(len(runner.memory()), len(runner.wrapper.action_table)).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.rounds = 0

    def rollout(self, start: Start) -> Rollout:
        weights, collectors = numpy_weights(self.net), self.collectors
        jobs = [
            collectors.pool.submit(
                collect_rollout,
                collectors.wrapper,
                collectors.config,
                start,
                weights,
                ROLLOUT_DECISIONS,
                seed,
                collectors.live,
            )
            for seed in range(self.rounds * collectors.emulators, (self.rounds + 1) * collectors.emulators)
        ]
        return merged([job.result() for job in jobs])

    def train_round(self) -> bool:
        start = load_policy_start(self.output / POLICY_START_FILE)
        if start is None:
            return False
        rollout = self.rollout(start)
        loss = update(self.net, self.optimizer, rollout, self.device)
        export_policy(self.net, self.output / POLICY_FILE)
        self.rounds += 1
        if rollout.exits and save_policy_exit(
            self.output / POLICY_EXITS_FILE, start.state, min(rollout.exits, key=len)
        ):
            logger.info(f"The policy found a way out of {start.state}; the search replays it next")
        episodes = max(1, sum(rollout.dones))
        exits, emulators = len(rollout.exits), self.collectors.emulators
        logger.info(
            f"Policy round {self.rounds} in {start.state}: reward {sum(rollout.rewards) / episodes:.2f} per episode, "
            f"{exits} exits in {episodes} episodes from {emulators} emulators, loss {loss:.3f}"
        )
        return True


def train_policy_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    logger.add(tas_dir(settings) / POLICY_LOG)
    emulators = settings.speedrun.policy_emulators
    with worker_pool(emulators) as pool:
        runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, os.getpid()))
        collectors = Collectors(wrapper, config, pool, emulators, tas_dir(settings) / LIVE_DIR)
        trainer = PolicyTrainer(runner, tas_dir(settings), gpu(), collectors)
        while multiprocessing.parent_process().is_alive():
            if not trainer.train_round():
                time.sleep(WAIT_SECONDS)
    logger.warning("The speedrun has stopped, so the policy stops with it")
