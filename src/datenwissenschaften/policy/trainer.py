import os
import time
from pathlib import Path

import torch
from loguru import logger

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.gpu import gpu
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.policy.network import PolicyNet, export_policy
from datenwissenschaften.policy.ppo import collect, update
from datenwissenschaften.policy.prior import POLICY_FILE
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.paths import tas_dir
from datenwissenschaften.tas.policy_start import POLICY_START_FILE, load_policy_start
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.workflow.wrapper import wrapper_class

LEARNING_RATE = 3e-4
WAIT_SECONDS = 30.0
POLICY_LOG = "policy.log"


class PolicyTrainer:
    def __init__(self, runner: SegmentRunner, output: Path, device: torch.device) -> None:
        self.runner = runner
        self.output = output
        self.device = device
        runner.power_on()
        self.net = PolicyNet(len(runner.memory()), len(runner.wrapper.action_table)).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        self.start: Start | None = None
        self.novelty = RamNovelty(runner.memory())
        self.rounds = 0

    def train_round(self) -> bool:
        start = load_policy_start(self.output / POLICY_START_FILE)
        if start is None:
            return False
        if start != self.start:
            self.runner.restore(start)
            self.start, self.novelty = start, RamNovelty(self.runner.memory())
        rollout = collect(self.runner, start, self.net, self.device, self.novelty)
        loss = update(self.net, self.optimizer, rollout, self.device)
        export_policy(self.net, self.output / POLICY_FILE)
        self.rounds += 1
        episodes = max(1, sum(rollout.dones))
        logger.info(
            f"Policy round {self.rounds} in {start.state}: reward {sum(rollout.rewards) / episodes:.2f} per episode, "
            f"{rollout.exits} exits in {episodes} episodes, loss {loss:.3f}"
        )
        return True


def train_policy_forever(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    logger.add(tas_dir(settings) / POLICY_LOG)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, os.getpid()))
    trainer = PolicyTrainer(runner, tas_dir(settings), gpu())
    while True:
        if not trainer.train_round():
            time.sleep(WAIT_SECONDS)
