import json
from pathlib import Path

import pytest
import torch
from fakes import ACTIONS, fake_decision, fake_laya_load, write_config

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.ppo import GAMMA, LAMBDA
from datenwissenschaften.rollout import Rollout
from datenwissenschaften.settings import load_config
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.state_models import StateModels

QUESTION = "Which move survives?"
OBSERVATION = {"state": json.dumps({"lives": 3}), "question": QUESTION}
EXPLORATION = {"Survive": 0.0, "Boss": 0.0}


@pytest.fixture
def models(tmp_path: Path, monkeypatch) -> StateModels:
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    agent = LayaAgent(LayaNetwork("fake/laya", ACTIONS, "cpu"), (QUESTION,))
    return StateModels(agent, RunContext(load_config(write_config(tmp_path)), "Level1"), ("Survive", "Boss"))


def sharpen(models: StateModels, state_name: str) -> None:
    models.activate(state_name)
    with torch.no_grad():
        for parameter in models.agent.policy.heads.policy.parameters():
            parameter.mul_(50.0)


def test_each_state_keeps_its_own_policy_and_reloads_it_from_disk(models: StateModels, tmp_path: Path):
    sharpen(models, "Survive")
    survivor = models.agent.act(OBSERVATION, 0.0, None).probabilities
    models.save()
    models.close()
    fresh = StateModels(models.agent, models.context, models.state_names)

    fresh.activate("Survive")

    assert fresh.agent.act(OBSERVATION, 0.0, None).probabilities == pytest.approx(survivor)
    fresh.activate("Boss")
    assert fresh.agent.act(OBSERVATION, 0.0, None).probabilities != pytest.approx(survivor)


def test_one_read_decides_for_every_emulator_with_its_state_policy(models: StateModels):
    sharpen(models, "Survive")
    survivor = models.agent.act(OBSERVATION, 0.0, None).probabilities
    models.activate("Boss")
    boss = models.agent.act(OBSERVATION, 0.0, None).probabilities

    decisions = models.decide([OBSERVATION] * 3, ["Survive", "Boss", "Survive"], EXPLORATION, [None] * 3)

    assert [decision.probabilities for decision in decisions] == [
        pytest.approx(survivor),
        pytest.approx(boss),
        pytest.approx(survivor),
    ]


def test_a_state_learns_from_the_rollouts_of_all_emulators(models: StateModels):
    for environment in range(3):
        models.rollout("Survive", environment).add(fake_decision(0, {"left": 1.0}, 1.0), 1.0, False, False)
    models.rollout("Boss", 0).add(fake_decision(0, {"left": 1.0}, 1.0), 1.0, False, False)
    models.activate("Survive")
    decision = models.agent.act(OBSERVATION, 0.0, None)
    for environment in range(3):
        models.rollouts[("Survive", environment)].decisions[0] = decision

    models.learn("Survive", [])

    assert models.agent.num_timesteps == 3
    assert models.collected("Survive") == 0
    assert models.collected("Boss") == 1


def test_advantages_never_run_across_emulators():
    empty = torch.zeros(2, 4)
    parts = [Rollout(), Rollout()]
    parts[0].add(Decision(0, {"left": 1.0}, 1.0, 0.5, empty, torch.zeros(4), None), 1.0, False, False)
    parts[1].add(Decision(0, {"left": 1.0}, 1.0, 0.25, empty, torch.zeros(4), None), 0.0, False, False)

    advantages, _ = Rollout.joined(parts).advantages(GAMMA, LAMBDA)

    assert advantages.tolist() == pytest.approx([1.0 + GAMMA * 0.5 - 0.5, GAMMA * 0.25 - 0.25])
