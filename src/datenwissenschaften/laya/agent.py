from pathlib import Path
from typing import Any

import torch

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.learning import GroupRelativeLearner
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.rollout import Rollout

Observation = dict[str, str]


class LayaAgent:
    def __init__(self, network: LayaNetwork, questions: tuple[str, ...]) -> None:
        for question in questions:
            network.question.require_fits(question)
        self.network = network
        self.learner = GroupRelativeLearner(network)
        self.num_timesteps = 0
        self.last_update: dict[str, float] = {}
        network.eval()

    @torch.no_grad()
    def act(self, observation: Observation, exploration: float) -> Decision:
        probabilities = torch.softmax(self.network([observation["state"]], [observation["question"]])[0], -1)
        behavior = (1 - exploration) * probabilities + exploration / len(probabilities)
        action = int(torch.distributions.Categorical(probs=behavior).sample())
        options = dict(zip(self.network.question.options, probabilities.tolist(), strict=True))
        return Decision(action, options, float(behavior[action]))

    def learn(self, rollout: Rollout) -> None:
        self.last_update = self.learner.update(rollout)
        self.num_timesteps += len(rollout)

    def restart(self) -> None:
        self.network.restore_pretrained()
        self.learner = GroupRelativeLearner(self.network)
        self.num_timesteps = 0
        self.last_update = {}

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save(
            {
                "network": self.network.state_dict(),
                "learner": self.learner.state_dict(),
                "num_timesteps": self.num_timesteps,
                "last_update": self.last_update,
            },
            temporary,
        )
        temporary.replace(path)

    def load(self, path: Path) -> None:
        saved = torch.load(path, map_location="cpu")
        self.network.load_state_dict(saved["network"])
        self.learner.load_state_dict(saved["learner"])
        self.num_timesteps = int(saved["num_timesteps"])
        self.last_update = saved["last_update"]

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.network.checkpoint,
            "actions": self.network.question.options,
            "device": str(self.network.device),
            "parameters": sum(parameter.numel() for parameter in self.network.parameters()),
            "num_timesteps": self.num_timesteps,
            "precision": str(self.network.dtype).removeprefix("torch."),
            "minibatch_size": self.learner.minibatch_size,
            **self.last_update,
        }
