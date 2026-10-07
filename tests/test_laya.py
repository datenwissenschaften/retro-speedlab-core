import io
import json

import pytest
import torch
from fakes import ACTIONS, FakeTokenizer, fake_laya_load

from datenwissenschaften.laya import network as network_module
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.ppo import GAMMA, LAMBDA
from datenwissenschaften.laya.question import LayaQuestion
from datenwissenschaften.laya.rollout import Rollout

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
    return agent.act(OBSERVATION, 0.0).probabilities["right"]


def rewarded_rollout(agent: LayaAgent, reward: dict[int, float]) -> Rollout:
    rollout = Rollout()
    for step in range(DECISIONS):
        decision = agent.act(OBSERVATION, 0.5)
        rollout.add(decision, reward[decision.action], step % 8 == 7)
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

    decision = agent.act(OBSERVATION, EXPLORATION)

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
    assert agent.act(OBSERVATION, 0.0).value > 0.1
    assert agent.num_timesteps == UPDATES * DECISIONS
    assert set(agent.last_update) == {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "imitation_loss",
        "demonstration_decisions",
    }
    assert all(not parameter.requires_grad for parameter in network.parameters())


def test_demonstrations_pull_the_policy_toward_their_moves(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    before = right_probability(agent)
    demonstrations = [DemonstrationStep(OBSERVATION["state"], QUESTION, 1)] * 8

    for _ in range(UPDATES):
        agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 0.0}), demonstrations)

    assert right_probability(agent) > before + 0.1
    assert agent.last_update["imitation_loss"] > 0.0
    assert agent.last_update["demonstration_decisions"] == len(demonstrations)


def test_advantages_follow_gae_and_stop_at_segment_ends():
    rollout = Rollout()
    empty = torch.zeros(2, 4)
    for value, reward, done in ((0.5, 1.0, False), (0.25, 0.0, True), (0.0, 2.0, False)):
        rollout.add(Decision(0, {"left": 1.0, "right": 0.0}, 1.0, value, empty, torch.zeros(4)), reward, done)

    advantages, returns = rollout.advantages(GAMMA, LAMBDA)

    last = 2.0 + GAMMA * 0.0 - 0.0
    second = 0.0 - 0.25
    first = (1.0 + GAMMA * 0.25 - 0.5) + GAMMA * LAMBDA * second
    assert advantages.tolist() == pytest.approx([first, second, last])
    assert returns.tolist() == pytest.approx([first + 0.5, second + 0.25, last])


def test_checkpoints_hold_the_heads_and_restart_returns_to_laya(network: LayaNetwork):
    agent = LayaAgent(network, (QUESTION,))
    pretrained = right_probability(agent)
    agent.learn(rewarded_rollout(agent, {0: 0.0, 1: 1.0}), [])
    trained = right_probability(agent)
    checkpoint = torch.load(io.BytesIO(agent.checkpoint().getvalue()), map_location="cpu")

    agent.restart()
    assert right_probability(agent) == pytest.approx(pretrained)
    agent.restore(checkpoint)

    assert right_probability(agent) == pytest.approx(trained)
    assert agent.num_timesteps == DECISIONS
    assert agent.metadata()["trained_parameters"] < agent.metadata()["reader_parameters"] * 10
