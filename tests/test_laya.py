import json
import math
from pathlib import Path

import pytest
import torch
from fakes import ACTIONS, FakeTokenizer, fake_laya_load

from datenwissenschaften.laya import learning as learning_module
from datenwissenschaften.laya import network as network_module
from datenwissenschaften.laya.agent import EXPLORATION_DECISIONS, FINAL_EXPLORATION, INITIAL_EXPLORATION, LayaAgent
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.learning import MAX_BACKTRACKS
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.question import LayaQuestion
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.laya.trust_region import TARGET_KL
from datenwissenschaften.laya.weight_snapshot import WeightSnapshot

QUESTION = "Which move survives?"
VISIBLE_LEARNING_RATE = 1e-3
OBSERVATION = {"state": json.dumps({"lives": 3, "score": 1}), "question": QUESTION}


@pytest.fixture
def network(monkeypatch) -> LayaNetwork:
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(learning_module, "ENCODER_LEARNING_RATE", VISIBLE_LEARNING_RATE)
    monkeypatch.setattr(learning_module, "HEAD_LEARNING_RATE", VISIBLE_LEARNING_RATE)
    return LayaNetwork("fake/laya", ACTIONS, "cpu")


def _rollout(steps: int) -> Rollout:
    rollout = Rollout()
    for step in range(steps):
        rollout.add(
            OBSERVATION["state"], QUESTION, Decision(step % 2, {"left": 0.5, "right": 0.5}, 0.5), float(step % 2), False
        )
    return rollout


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

    assert agent.num_timesteps == len(rollout)
    after = list(network.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))
    assert set(agent.last_update) == {
        "policy_loss",
        "entropy",
        "step_kl",
        "kl",
        "learning_rate_scale",
    }
    assert agent.last_update["kl"] >= 0.0
    assert agent.metadata()["entropy"] == agent.last_update["entropy"]


def test_agent_checkpoint_round_trip(network: LayaNetwork, monkeypatch, tmp_path: Path):
    agent = LayaAgent(network, (QUESTION,))
    agent.num_timesteps = 42
    agent.learner.trust_region.learning_rate_scale = 0.3
    agent.last_update = {"kl": 0.01}
    path = tmp_path / "laya.pt"
    agent.save(path)
    restored = LayaAgent(LayaNetwork("fake/laya", ACTIONS, "cpu"), (QUESTION,))

    restored.load(path)

    assert restored.num_timesteps == 42
    assert restored.learner.trust_region.learning_rate_scale == 0.3
    assert restored.last_update == {"kl": 0.01}
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


def test_float16_learning_scales_gradients_and_stays_finite(monkeypatch):
    monkeypatch.setattr(network_module, "autocast_dtype", lambda device: torch.float16)
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    monkeypatch.setattr(learning_module, "HEAD_LEARNING_RATE", VISIBLE_LEARNING_RATE)
    network = LayaNetwork("fake/laya", ACTIONS, "cpu")
    agent = LayaAgent(network, (QUESTION,))
    before = [parameter.detach().clone() for parameter in network.parameters()]
    rollout = Rollout()
    for step in range(12):
        rollout.add(
            OBSERVATION["state"], QUESTION, Decision(step % 2, {"left": 0.5, "right": 0.5}, 0.5), 1.0, step == 5
        )

    agent.learn(rollout)

    assert agent.learner.scaler.is_enabled()
    assert agent.metadata()["precision"] == "float16"
    assert all(torch.isfinite(parameter).all() for parameter in network.parameters())
    assert any(not torch.equal(old, new) for old, new in zip(before, network.parameters(), strict=True))


def test_agent_restart_returns_to_the_pretrained_laya(network: LayaNetwork):
    pretrained = [parameter.detach().clone() for parameter in network.parameters()]
    agent = LayaAgent(network, (QUESTION,))
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(1.0)
    agent.num_timesteps, agent.last_update = 99, {"kl": 1.0}

    agent.restart()

    assert (agent.num_timesteps, agent.last_update) == (0, {})
    assert all(torch.equal(old, new) for old, new in zip(pretrained, network.parameters(), strict=True))


def test_an_overshooting_update_is_pulled_back_into_the_trust_region(network: LayaNetwork):
    learner = LayaAgent(network, (QUESTION,)).learner
    measured, blends = iter([0.5, 0.001]), []
    learner._measure_kl = lambda rollout, previous: next(measured)
    learner.snapshot.blend = blends.append

    kl = learner._backtrack(_rollout(2), torch.ones(2, 2), 1.0)

    assert kl == 0.001
    assert blends == pytest.approx([math.sqrt(TARGET_KL / 1.0), math.sqrt(TARGET_KL / 0.5)])


def test_an_update_that_stays_too_far_away_is_reverted(network: LayaNetwork):
    learner = LayaAgent(network, (QUESTION,)).learner
    blends = []
    learner._measure_kl = lambda rollout, previous: 0.0 if blends[-1:] == [0.0] else 1.0
    learner.snapshot.blend = blends.append

    kl = learner._backtrack(_rollout(2), torch.ones(2, 2), 1.0)

    assert kl == 0.0
    assert len(blends) == MAX_BACKTRACKS + 1
    assert blends[-1] == 0.0


def test_weight_snapshot_blends_back_toward_the_captured_weights():
    parameter = torch.nn.Parameter(torch.zeros(3))
    snapshot = WeightSnapshot([parameter])
    snapshot.capture()
    with torch.no_grad():
        parameter.add_(4.0)

    snapshot.blend(0.25)

    assert torch.equal(parameter.detach(), torch.full((3,), 1.0))
