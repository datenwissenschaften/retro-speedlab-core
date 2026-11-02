import random
from pathlib import Path

import msgspec
import numpy as np
import pytest
import torch
from fakes import saving_environment
from test_tas import START, LineRunner

from datenwissenschaften.policy import ppo
from datenwissenschaften.policy.network import PolicyNet, export_policy
from datenwissenschaften.policy.prior import PolicyPrior
from datenwissenschaften.policy.trainer import PolicyTrainer
from datenwissenschaften.tas.policy_start import POLICY_START_FILE, save_policy_start
from datenwissenschaften.tas.preferences import Preferences
from datenwissenschaften.tas.prior import ActionPrior
from datenwissenschaften.tas.runner import SegmentRunner

CPU = torch.device("cpu")


def test_the_workers_read_the_same_policy_the_trainer_exported(tmp_path: Path):
    net = PolicyNet(4, 3)
    memory = np.asarray([0, 64, 128, 255], dtype=np.uint8)
    export_policy(net, tmp_path / "policy.npz")

    probabilities = PolicyPrior(tmp_path / "policy.npz").probabilities(memory, 3)

    with torch.no_grad():
        logits, _ = net(torch.from_numpy(memory.astype(np.float32) / 255)[None])
    assert np.allclose(probabilities, torch.softmax(logits[0], 0).numpy(), atol=1e-5)
    assert PolicyPrior(tmp_path / "none.npz").choose(memory, 3, random.Random(1)) is None


def test_ppo_plays_episodes_from_the_start_and_learns_from_them(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ppo, "ROLLOUT_DECISIONS", 256)
    runner, net = LineRunner(), PolicyNet(1, 3)
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    rollout = ppo.collect(runner, START, net, CPU, ppo.RamNovelty(runner.memory()))
    loss = ppo.update(net, optimizer, rollout, CPU)

    assert len(rollout.actions) == len(rollout.rewards) == 256 and any(rollout.dones)
    assert np.isfinite(loss)
    gains, returns = ppo.advantages(rollout)
    assert gains.shape == returns.shape == (256,)


def test_the_trainer_waits_for_a_level_and_then_trains_and_exports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ppo, "ROLLOUT_DECISIONS", 64)
    runner = SegmentRunner(saving_environment(tmp_path))
    trainer = PolicyTrainer(runner, tmp_path, CPU)

    assert not trainer.train_round()
    save_policy_start(tmp_path / POLICY_START_FILE, runner.power_on())
    assert trainer.train_round()
    assert PolicyPrior(tmp_path / "policy.npz").probabilities(runner.memory(), 2) is not None


def test_the_search_follows_the_table_or_the_policy_part_of_the_time(tmp_path: Path):
    (tmp_path / "prior.msgpack").write_bytes(msgspec.msgpack.encode({"Walk": {2: 1}}))
    export_policy(PolicyNet(1, 3), tmp_path / "policy.npz")
    preferences = Preferences(ActionPrior(tmp_path / "prior.msgpack"), PolicyPrior(tmp_path / "policy.npz"))
    rng = random.Random(3)

    chosen = [preferences.choose(LineRunner(), 3, rng) for _ in range(200)]

    assert chosen.count(None) > 50 and chosen.count(2) > 50
    assert {choice for choice in chosen if choice is not None} <= {0, 1, 2}
