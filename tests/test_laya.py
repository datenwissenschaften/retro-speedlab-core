import io
import json

import numpy as np
import pytest
import torch
from fakes import ACTIONS, FakeTokenizer, fake_laya_load

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.question import LayaQuestion
from datenwissenschaften.ppo import GAMMA, LAMBDA
from datenwissenschaften.rollout import Rollout

QUESTION = "Which move survives?"
EXPLORATION = 0.2
OBSERVATION = {"state": json.dumps({"lives": 3, "score": 1}), "question": QUESTION}
UPDATES = 60
DECISIONS = 64


@pytest.fixture
def network(monkeypatch) -> LayaNetwork:
    monkeypatch.setattr(network_module.laya, "load", fake_laya_load)
    return LayaNetwork("fake/laya", ACTIONS, "cpu")


def right_probability(agent: LayaAgent) -> float:
    return agent.act(OBSERVATION, 0.0, None).probabilities["right"]


def rewarded_rollout(agent: LayaAgent, reward: dict[int, float]) -> Rollout:
    rollout = Rollout()
    for step in range(DECISIONS):
        decision = agent.act(OBSERVATION, 0.5, None)
        rollout.add(decision, reward[decision.action], step % 8 == 7, False)
    return rollout


def test_laya_reads_the_game_frozen_and_returns_one_vector_per_option(network: LayaNetwork):
    options, pooled = network.features([OBSERVATION["state"]], [QUESTION])

    assert all(not parameter.requires_grad for parameter in network.parameters())
    assert options.shape == (1, len(ACTIONS), network.width)
    assert pooled.shape == (1, network.width)


def test_option_vectors_follow_the_action_order_whatever_order_laya_saw(network: LayaNetwork, monkeypatch):
    monkeypatch.setattr(network.question, "_order", lambda state, question: [1, 0])
    options, _ = network.features([OBSERVATION["state"]], [QUESTION])
    batch, _ = network.question.encode([OBSERVATION["state"]], [QUESTION], network.device)
    decision = network.decision
    hidden = decision.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
    hidden = hidden + decision.type_emb(batch["qtype"])[:, None, :]
    first_marker, second_marker = batch["marker_pos"][0].tolist()

    assert torch.allclose(options[0, 1], hidden[0, first_marker])
    assert torch.allclose(options[0, 0], hidden[0, second_marker])


def test_question_rejects_prompts_that_hide_options():
    question = LayaQuestion(FakeTokenizer(), {"max_len": 8, "head_max_len": 48}, ACTIONS)

    with pytest.raises(ValueError, match="budget"):
        question.require_fits(QUESTION)


def test_options_appear_in_a_stable_order_per_state_that_varies_across_states():
    question = LayaQuestion(FakeTokenizer(), {"max_len": 128, "head_max_len": 48}, dict.fromkeys("abcdef", "move"))

    orders = {tuple(question._order(f'{{"x": {x}}}', QUESTION)) for x in range(20)}

    assert question._order('{"x": 1}', QUESTION) == question._order('{"x": 1}', QUESTION)
    assert all(sorted(order) == list(range(6)) for order in orders)
    assert len(orders) > 1


def test_a_fresh_state_starts_with_lays_own_judgement(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    options, _ = network.features([OBSERVATION["state"]], [QUESTION])

    decision = agent.act(OBSERVATION, EXPLORATION, None)

    expected = torch.softmax(network.scorer(options).squeeze(-1)[0], -1)
    assert list(decision.probabilities.values()) == pytest.approx(expected.tolist())
    chosen = list(decision.probabilities.values())[decision.action]
    assert decision.behavior_probability == pytest.approx((1 - EXPLORATION) * chosen + EXPLORATION / len(ACTIONS))
    assert decision.value == 0.0
    assert decision.options.shape == (len(ACTIONS), network.width)


def test_ppo_makes_the_rewarded_move_likelier_and_learns_its_value(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    before = right_probability(agent)

    for _ in range(UPDATES):
        agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 1.0}), [])

    assert right_probability(agent) > before + 0.1
    assert agent.act(OBSERVATION, 0.0, None).value > 0.1
    assert agent.num_timesteps == UPDATES * DECISIONS
    assert set(agent.last_update) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "imitation_loss",
        "skipped_minibatches",
        "demonstration_decisions",
    }
    assert all(not parameter.requires_grad for parameter in network.parameters())


def test_demonstrations_pull_the_policy_toward_their_moves(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    before = right_probability(agent)
    demonstrations = [DemonstrationStep(OBSERVATION["state"], QUESTION, 1, np.zeros(2, dtype=np.float32))] * 8

    for _ in range(UPDATES):
        agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 0.0}), demonstrations)

    assert right_probability(agent) > before + 0.1
    assert agent.last_update["imitation_loss"] > 0.0
    assert agent.last_update["demonstration_decisions"] == len(demonstrations)


def decision_valued(value: float) -> Decision:
    return Decision(0, {"left": 1.0, "right": 0.0}, 1.0, 1.0, value, torch.zeros(2, 4), torch.zeros(4), None)


def test_advantages_follow_gae_and_stop_at_segment_ends():
    rollout = Rollout()
    for value, reward, terminal in ((0.5, 1.0, False), (0.25, 0.0, True), (0.0, 2.0, False)):
        rollout.add(decision_valued(value), reward, terminal, False)

    advantages, returns = rollout.advantages(GAMMA, LAMBDA)

    last = 2.0 + GAMMA * 0.0 - 0.0
    second = 0.0 - 0.25
    first = (1.0 + GAMMA * 0.25 - 0.5) + GAMMA * LAMBDA * second
    assert advantages.tolist() == pytest.approx([first, second, last])
    assert returns.tolist() == pytest.approx([first + 0.5, second + 0.25, last])


def test_a_truncated_segment_bootstraps_from_its_value_instead_of_ending_the_game():
    rollout = Rollout()
    rollout.add(decision_valued(0.5), 1.0, False, True)
    rollout.add(decision_valued(0.25), 0.0, True, False)

    advantages, _ = rollout.advantages(GAMMA, LAMBDA)

    assert advantages.tolist() == pytest.approx([1.0 + GAMMA * 0.5 - 0.5, -0.25])


def test_checkpoints_hold_the_heads_and_restart_returns_to_laya(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    pretrained = right_probability(agent)
    agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 1.0}), [])
    trained = right_probability(agent)
    checkpoint = torch.load(io.BytesIO(agent.policy.checkpoint().getvalue()), map_location="cpu")

    agent.restart()
    assert right_probability(agent) == pytest.approx(pretrained)
    agent.policy.restore(checkpoint)

    assert right_probability(agent) == pytest.approx(trained)
    assert agent.num_timesteps == DECISIONS
    assert agent.metadata()["trained_parameters"] < agent.metadata()["reader_parameters"] * 10


def test_a_checkpoint_with_a_broken_policy_head_restores_laya_judgement_and_keeps_the_value_head(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    pretrained = right_probability(agent)
    agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 1.0}), [])
    checkpoint = torch.load(io.BytesIO(agent.policy.checkpoint().getvalue()), map_location="cpu")
    value = {name: tensor.clone() for name, tensor in checkpoint["heads"].items() if name.startswith("value.")}
    for name, tensor in checkpoint["heads"].items():
        if name.startswith("policy."):
            tensor.fill_(float("nan"))

    agent.restart()
    agent.policy.restore(checkpoint)

    assert right_probability(agent) == pytest.approx(pretrained)
    assert all(torch.equal(agent.policy.heads.state_dict()[name].cpu(), tensor) for name, tensor in value.items())
    assert agent.num_timesteps == DECISIONS


def test_laya_reports_how_often_its_favourite_move_is_the_advised_one(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    rollout = Rollout()
    for advice in (0, 1, None, 1):
        decision = agent.act(OBSERVATION, 0.0, advice)
        rollout.add(decision, 0.0, False, False)
    favourite = max(rollout.decisions[0].probabilities, key=rollout.decisions[0].probabilities.__getitem__)
    expected = sum(favourite == name for name in ("left", "right", "right")) / 3

    agent.learn(rollout, [])

    assert agent.last_update["advisor_agreement"] == pytest.approx(expected)


def test_without_advice_there_is_no_agreement_to_report(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 1.0}), [])

    assert "advisor_agreement" not in agent.last_update
