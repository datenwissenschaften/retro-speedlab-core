import random
from typing import Any

import torch

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.heads import Heads
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.policy import Policy
from datenwissenschaften.laya.ppo import MINIBATCH, Demonstration, PpoLearner
from datenwissenschaften.laya.rollout import Rollout

Observation = dict[str, str]
Features = tuple[torch.Tensor, torch.Tensor]


class LayaAgent:
    def __init__(self, network: LayaNetwork, questions: tuple[str, ...]) -> None:
        for question in questions:
            network.question.require_fits(question)
        self.network = network
        self.restart()

    @property
    def num_timesteps(self) -> int:
        return self.policy.num_timesteps

    @property
    def last_update(self) -> dict[str, float]:
        return self.policy.last_update

    def read(self, observations: list[Observation]) -> Features:
        states = [observation["state"] for observation in observations]
        return self.network.features(states, [observation["question"] for observation in observations])

    @torch.no_grad()
    def decide(self, features: Features, exploration: float) -> list[Decision]:
        options, pooled = features
        probabilities = torch.softmax(self.policy.heads.policy(options), -1)
        behavior = (1 - exploration) * probabilities + exploration / probabilities.size(-1)
        actions = torch.distributions.Categorical(probs=behavior).sample().tolist()
        values = self.policy.heads.value(pooled).tolist()
        names = self.network.question.options
        return [
            Decision(
                action,
                dict(zip(names, probabilities[row].tolist(), strict=True)),
                float(behavior[row, action]),
                values[row],
                options[row].cpu(),
                pooled[row].cpu(),
            )
            for row, action in enumerate(actions)
        ]

    def act(self, observation: Observation, exploration: float) -> Decision:
        return self.decide(self.read([observation]), exploration)[0]

    def learn(self, rollout: Rollout, demonstrations: list[DemonstrationStep]) -> None:
        self.policy.last_update = self.policy.learner.update(rollout, self._demonstration(demonstrations))
        self.policy.num_timesteps += len(rollout)

    def _demonstration(self, demonstrations: list[DemonstrationStep]) -> Demonstration:
        if not demonstrations:
            return None
        sample = random.sample(demonstrations, min(len(demonstrations), MINIBATCH))
        options, _ = self.network.features([step.state for step in sample], [step.question for step in sample])
        return options, torch.tensor([step.action for step in sample], device=options.device)

    def restart(self) -> None:
        heads = Heads(self.network.scorer, self.network.width).to(self.network.device).eval()
        self.policy = Policy(heads, PpoLearner(heads), 0, {})

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.network.checkpoint,
            "actions": self.network.question.options,
            "device": str(self.network.device),
            "reader_parameters": sum(parameter.numel() for parameter in self.network.parameters()),
            "trained_parameters": sum(parameter.numel() for parameter in self.policy.heads.parameters()),
            "num_timesteps": self.policy.num_timesteps,
            "precision": str(self.network.dtype).removeprefix("torch."),
            "minibatch_size": MINIBATCH,
            **self.policy.last_update,
        }
