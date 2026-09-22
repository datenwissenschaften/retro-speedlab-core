import numpy as np
import pytest
import torch
from gymnasium import spaces
from sb3_contrib.common.recurrent.type_aliases import RNNStates

from datenwissenschaften.segmented_rollout import SegmentedRecurrentRollouts


class FakePolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.training_mode_calls: list[bool] = []

    def set_training_mode(self, mode):
        self.training_mode_calls.append(mode)

    def __call__(self, obs, states, starts):
        batch = starts.shape[0]
        if isinstance(self.action_space, spaces.Box):
            actions = torch.full((batch, *self.action_space.shape), 100.0)
        else:
            actions = torch.zeros(batch, dtype=torch.int64)
        values = torch.arange(batch, dtype=torch.float32).reshape(batch, 1)
        log_probs = torch.zeros(batch)
        new_states = _make_states(batch)
        return actions, values, log_probs, new_states

    def predict_values(self, obs, vf_states, starts):
        return torch.tensor([[1.5]])


def _make_states(batch: int) -> RNNStates:
    zeros = torch.zeros((1, batch, 1))
    return RNNStates(pi=(zeros, zeros), vf=(zeros, zeros))


def _model(*, action_space, rnd=None):
    return type(
        "Model",
        (),
        {
            "policy": FakePolicy(action_space),
            "action_space": action_space,
            "device": torch.device("cpu"),
            "rnd": rnd,
            "n_steps": 100,
            "n_envs": 2,
            "_last_lstm_states": _make_states(2),
        },
    )()


def _observations(num_envs: int) -> dict[str, np.ndarray]:
    return {"value": np.arange(num_envs, dtype=np.float32).reshape(num_envs, 1)}


def test_actions_selects_the_configured_policy_per_environment_and_clips_box_actions():
    model = _model(action_space=spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32))
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=2)

    actions, decisions = rollouts.actions(_observations(2), ["Find", "Find"])

    assert actions.shape == (2, 1)
    assert (actions <= 1.0).all()
    assert model.policy.training_mode_calls == [False]
    assert set(decisions) == {0, 1}


def test_actions_produces_discrete_actions_reshaped_to_one_dimension():
    model = _model(action_space=spaces.Discrete(4))
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=2)

    actions, decisions = rollouts.actions(_observations(2), ["Find", "Find"])

    assert actions.shape == (2,)
    assert set(decisions) == {0, 1}


def test_actions_routes_different_environments_to_different_state_models():
    find_model = _model(action_space=spaces.Discrete(2))
    eat_model = _model(action_space=spaces.Discrete(2))
    rollouts = SegmentedRecurrentRollouts({"Find": find_model, "Eat": eat_model}, num_envs=2)

    rollouts.actions(_observations(2), ["Find", "Eat"])

    assert find_model.policy.training_mode_calls == [False]
    assert eat_model.policy.training_mode_calls == [False]


def _decisions_for(rollouts: SegmentedRecurrentRollouts, state_names: list[str]):
    observations = _observations(len(state_names))
    actions, decisions = rollouts.actions(observations, state_names)
    return observations, actions, decisions


def test_append_records_a_transition_and_reports_when_the_rollout_is_full():
    model = _model(action_space=spaces.Discrete(2))
    model.n_steps = 1
    model.n_envs = 1
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)
    observations, actions, decisions = _decisions_for(rollouts, ["Find"])
    new_observations = _observations(1)

    full = rollouts.append(
        observations,
        actions,
        rewards=np.asarray([1.0]),
        new_observations=new_observations,
        dones=np.asarray([False]),
        infos=[{}],
        state_names=["Find"],
        decisions=decisions,
    )

    assert full == {"Find"}
    assert len(rollouts.transitions["Find"]) == 1
    assert rollouts.transitions["Find"][0].segment_end is False


def test_append_ends_the_segment_and_resets_on_done():
    model = _model(action_space=spaces.Discrete(2))
    model.n_steps = 100
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)
    observations, actions, decisions = _decisions_for(rollouts, ["Find"])
    new_observations = _observations(1)

    full = rollouts.append(
        observations,
        actions,
        rewards=np.asarray([1.0]),
        new_observations=new_observations,
        dones=np.asarray([True]),
        infos=[{}],
        state_names=["Find"],
        decisions=decisions,
    )

    assert full == set()
    transition = rollouts.transitions["Find"][0]
    assert transition.segment_end is True
    assert transition.next_value == 0.0
    assert rollouts.episode_starts["Find"][0] == np.True_


def test_append_skips_environments_outside_the_enabled_states_but_resets_them_on_done():
    model = _model(action_space=spaces.Discrete(2))
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)
    observations, actions, decisions = _decisions_for(rollouts, ["Find"])
    new_observations = _observations(1)

    full = rollouts.append(
        observations,
        actions,
        rewards=np.asarray([1.0]),
        new_observations=new_observations,
        dones=np.asarray([True]),
        infos=[{}],
        state_names=["Find"],
        decisions=decisions,
        enabled_states=set(),
    )

    assert full == set()
    assert rollouts.transitions["Find"] == []
    assert rollouts.episode_starts["Find"][0] == np.True_


def test_append_adds_intrinsic_reward_when_the_model_has_rnd():
    rnd = type(
        "FakeRND",
        (),
        {"intrinsic_reward": lambda self, obs, dones: (np.asarray([2.0]), 0.5)},
    )()
    model = _model(action_space=spaces.Discrete(2), rnd=rnd)
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)
    observations, actions, decisions = _decisions_for(rollouts, ["Find"])
    new_observations = _observations(1)

    rollouts.append(
        observations,
        actions,
        rewards=np.asarray([1.0]),
        new_observations=new_observations,
        dones=np.asarray([False]),
        infos=[{}],
        state_names=["Find"],
        decisions=decisions,
    )

    transition = rollouts.transitions["Find"][0]
    assert transition.reward == 1.0 + 0.5 * 2.0


def test_build_buffer_rejects_a_state_with_no_collected_transitions():
    model = _model(action_space=spaces.Discrete(2))
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)

    with pytest.raises(ValueError, match="Find"):
        rollouts.build_buffer("Find")


def test_reset_environment_zeroes_lstm_state_and_marks_episode_start():
    model = _model(action_space=spaces.Discrete(2))
    rollouts = SegmentedRecurrentRollouts({"Find": model}, num_envs=1)
    rollouts.episode_starts["Find"][0] = False
    rollouts.lstm_states["Find"].pi[0][:, 0, :] = 5.0

    rollouts.reset_environment(0)

    assert rollouts.episode_starts["Find"][0] == np.True_
    assert torch.equal(rollouts.lstm_states["Find"].pi[0][:, 0, :], torch.zeros(1, 1))
