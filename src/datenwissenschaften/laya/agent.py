import io
import random
from typing import Any

import torch

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.heads import Heads
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.ppo import MINIBATCH, Demonstration, PpoLearner
from datenwissenschaften.laya.rollout import Rollout

Observation = dict[str, str]


class LayaAgent:
    def __init__(self, network: LayaNetwork, questions: tuple[str, ...]) -> None:
        for question in questions:
            network.question.require_fits(question)
        self.network = network
        self.restart()

    @torch.no_grad()
    def act(self, observation: Observation, exploration: float) -> Decision:
        options, pooled = self.network.features([observation["state"]], [observation["question"]])
        probabilities = torch.softmax(self.heads.policy(options)[0], -1)
        behavior = (1 - exploration) * probabilities + exploration / len(probabilities)
        action = int(torch.distributions.Categorical(probs=behavior).sample())
        return Decision(
            action,
            dict(zip(self.network.question.options, probabilities.tolist(), strict=True)),
            float(behavior[action]),
            float(self.heads.value(pooled)[0]),
            options[0].cpu(),
            pooled[0].cpu(),
        )

    def learn(self, rollout: Rollout, demonstrations: list[DemonstrationStep]) -> None:
        self.last_update = self.learner.update(rollout, self._demonstration(demonstrations))
        self.num_timesteps += len(rollout)

    def _demonstration(self, demonstrations: list[DemonstrationStep]) -> Demonstration:
        if not demonstrations:
            return None
        sample = random.sample(demonstrations, min(len(demonstrations), MINIBATCH))
        options, _ = self.network.features([step.state for step in sample], [step.question for step in sample])
        return options, torch.tensor([step.action for step in sample], device=options.device)

    def restart(self) -> None:
        self.heads = Heads(self.network.scorer, self.network.width).to(self.network.device).eval()
        self.learner = PpoLearner(self.heads)
        self.num_timesteps = 0
        self.last_update: dict[str, float] = {}

    def checkpoint(self) -> io.BytesIO:
        buffer = io.BytesIO()
        torch.save(
            {
                "heads": self.heads.state_dict(),
                "learner": self.learner.state_dict(),
                "num_timesteps": self.num_timesteps,
                "last_update": self.last_update,
            },
            buffer,
        )
        return buffer

    def restore(self, checkpoint: dict[str, Any]) -> None:
        self.heads.load_state_dict(checkpoint["heads"])
        self.learner.load_state_dict(checkpoint["learner"])
        self.num_timesteps = int(checkpoint["num_timesteps"])
        self.last_update = checkpoint["last_update"]

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.network.checkpoint,
            "actions": self.network.question.options,
            "device": str(self.network.device),
            "reader_parameters": sum(parameter.numel() for parameter in self.network.parameters()),
            "trained_parameters": sum(parameter.numel() for parameter in self.heads.parameters()),
            "num_timesteps": self.num_timesteps,
            "precision": str(self.network.dtype).removeprefix("torch."),
            "minibatch_size": MINIBATCH,
            **self.last_update,
        }
