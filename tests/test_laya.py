import json
from pathlib import Path

import pytest
import torch
from fakes import ACTIONS, FakeTokenizer, fake_laya_load

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.laya.agent import EXPLORATION_DECISIONS, FINAL_EXPLORATION, INITIAL_EXPLORATION, LayaAgent
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.question import LayaQuestion
from datenwissenschaften.laya.rollout import Rollout

QUESTION = "Which move survives?"
OBSERVATION = {"state": json.dumps({"lives": 3, "score": 1}), "question": QUESTION}


@pytest.fixture
def network(monkeypatch) -> LayaNetwork:
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    return LayaNetwork("fake/laya", ACTIONS, "cpu")


def test_network_enables_memory_saving_and_scores_every_option(network: LayaNetwork):
    logits = network([OBSERVATION["state"]], [QUESTION])

    assert network.decision.encoder.checkpointing is True
    assert network.decision.head_checkpointing is True
    assert logits.shape == (1, len(ACTIONS))
    assert set(map(id, network.encoder_parameters())).isdisjoint(map(id, network.head_parameters()))


def test_question_rejects_prompts_that_hide_options():
    question = LayaQuestion(FakeTokenizer(), {"max_len": 8, "head_max_len": 48}, ACTIONS)

    with pytest.raises(ValueError, match="budget"):
        question.require_fits(QUESTION)


def test_agent_decision_carries_every_action_probability(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))

    decision = agent.act(OBSERVATION)

    assert decision.action in range(len(ACTIONS))
    assert list(decision.probabilities) == list(ACTIONS)
    assert sum(decision.probabilities.values()) == pytest.approx(1.0)
    chosen = list(decision.probabilities.values())[decision.action]
    assert decision.behavior_probability == pytest.approx(
        (1 - INITIAL_EXPLORATION) * chosen + INITIAL_EXPLORATION / len(ACTIONS)
    )


def test_agent_learning_changes_every_trainable_part(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    before = [parameter.detach().clone() for parameter in network.parameters()]
    rollout = Rollout()
    for step in range(20):
        rollout.add(
            OBSERVATION["state"],
            QUESTION,
            Decision(step % 2, {"left": 0.5, "right": 0.5}, 0.5),
            float(step % 2),
            step == 9,
        )

    agent.learn(rollout)

    after = list(network.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert set(agent.last_update) == {"policy_loss", "entropy", "kl", "entropy_coefficient", "learning_rate_scale"}
    assert agent.last_update["kl"] >= 0.0
    assert agent.metadata()["entropy"] == agent.last_update["entropy"]


def test_agent_checkpoint_round_trip(network: LayaNetwork, monkeypatch, tmp_path: Path):
    agent = LayaAgent(network, (QUESTION,))
    agent.num_timesteps = 42
    path = tmp_path / "laya.pt"
    agent.save(path)
    restored = LayaAgent(LayaNetwork("fake/laya", ACTIONS, "cpu"), (QUESTION,))

    restored.load(path)

    assert restored.num_timesteps == 42
    assert restored.metadata()["actions"] == ACTIONS
    for original, loaded in zip(network.parameters(), restored.network.parameters(), strict=True):
        assert torch.equal(original, loaded)


def test_rollout_advantages_are_normalized_and_cut_at_episode_ends():
    rollout = Rollout()
    for reward, done in ((1.0, True), (0.0, False), (0.0, False)):
        rollout.add("{}", QUESTION, Decision(0, {"left": 1.0, "right": 0.0}, 0.9), reward, done)

    advantages = rollout.group_relative_advantages(0.9)

    assert len(rollout) == 3
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-6)
    assert advantages[0] > advantages[1] == advantages[2]


def test_exploration_fades_as_laya_gains_experience(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    start = agent.exploration
    agent.num_timesteps = EXPLORATION_DECISIONS // 2
    halfway = agent.exploration
    agent.num_timesteps = EXPLORATION_DECISIONS * 3

    assert start == INITIAL_EXPLORATION
    assert FINAL_EXPLORATION < halfway < INITIAL_EXPLORATION
    assert agent.exploration == pytest.approx(FINAL_EXPLORATION)
    assert agent.metadata()["exploration"] == FINAL_EXPLORATION
