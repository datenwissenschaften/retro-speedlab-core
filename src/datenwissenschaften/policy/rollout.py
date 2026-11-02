from dataclasses import dataclass

import numpy as np

from datenwissenschaften.policy.prior import (
    VALUE_BIAS,
    VALUE_WEIGHT,
    Weights,
    hidden_of,
    layers_of,
    log_probabilities,
    observation,
)
from datenwissenschaften.tas.live import WorkerLive
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import leaves

ROLLOUT_DECISIONS = 1024
EPISODE_DECISIONS = 128
NOVELTY_REWARD = 0.01
EXIT_REWARD = 10.0
DEATH_PENALTY = 1.0


@dataclass(slots=True)
class Rollout:
    observations: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]
    exits: int


def reward_of(outcome_reward: float, novelty: int, exited: bool, alive: bool) -> float:
    return outcome_reward + NOVELTY_REWARD * novelty + EXIT_REWARD * exited - DEATH_PENALTY * (not alive)


def collect(
    runner: SegmentRunner,
    start: Start,
    weights: Weights,
    novelty: RamNovelty,
    decisions: int,
    rng: np.random.Generator,
    live: WorkerLive | None,
) -> Rollout:
    layers, actions = layers_of(weights), len(runner.wrapper.action_table)
    rollout = Rollout([], [], [], [], [], [], 0)
    runner.restore(start)
    played = 0
    while len(rollout.actions) < decisions:
        memory = runner.memory()
        hidden = hidden_of(layers, memory)
        log_probs = log_probabilities(layers, hidden, actions)
        action = int(rng.choice(actions, p=np.exp(log_probs) / np.exp(log_probs).sum()))
        outcome = runner.decide(action)
        played += 1
        if live is not None:
            live.step(runner, sum(rollout.dones), None, len(outcome.pressed))
        exited = leaves(runner, start, outcome.state, outcome.won)
        ended = exited or not outcome.alive or outcome.state != start.state or played >= EPISODE_DECISIONS
        rollout.observations.append(observation(memory))
        rollout.actions.append(action)
        rollout.log_probs.append(float(log_probs[action]))
        rollout.values.append(float(weights[VALUE_WEIGHT] @ hidden + weights[VALUE_BIAS]))
        rollout.rewards.append(reward_of(outcome.reward, novelty.see(runner.memory()), exited, outcome.alive))
        rollout.dones.append(ended or len(rollout.actions) == decisions)
        rollout.exits += exited
        if ended:
            runner.restore(start)
            played = 0
    return rollout


def merged(rollouts: list[Rollout]) -> Rollout:
    return Rollout(
        [seen for rollout in rollouts for seen in rollout.observations],
        [action for rollout in rollouts for action in rollout.actions],
        [log_prob for rollout in rollouts for log_prob in rollout.log_probs],
        [value for rollout in rollouts for value in rollout.values],
        [reward for rollout in rollouts for reward in rollout.rewards],
        [done for rollout in rollouts for done in rollout.dones],
        sum(rollout.exits for rollout in rollouts),
    )
