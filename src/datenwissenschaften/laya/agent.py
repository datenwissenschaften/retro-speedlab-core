from pathlib import Path
from typing import Any

import torch

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.learning import GroupRelativeLearner
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.laya.rollout import Rollout

Observation = dict[str, str]

INITIAL_EXPLORATION = 0.2
FINAL_EXPLORATION = 0.05
EXPLORATION_DECISIONS = 50_000


class LayaAgent:
    def __init__(self, network: LayaNetwork, questions: tuple[str, ...]) -> None:
        for question in questions:
            network.question.require_fits(question)
        self.network = network
        self.learner = GroupRelativeLearner(network)
        self.num_timesteps = 0
        self.last_update: dict[str, float] = {}
        network.eval()

    @property
    def exploration(self) -> float:
        progress = min(1.0, self.num_timesteps / EXPLORATION_DECISIONS)
        return INITIAL_EXPLORATION + (FINAL_EXPLORATION - INITIAL_EXPLORATION) * progress

    @torch.no_grad()
    def act(self, observation: Observation) -> Decision:
        probabilities = torch.softmax(self.network([observation["state"]], [observation["question"]])[0], -1)
        behavior = (1 - self.exploration) * probabilities + self.exploration / len(probabilities)
        action = int(torch.distributions.Categorical(probs=behavior).sample())
        options = dict(zip(self.network.question.options, probabilities.tolist(), strict=True))
        return Decision(action, options, float(behavior[action]))

    def learn(self, rollout: Rollout) -> None:
        self.last_update = self.learner.update(rollout)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".tmp")
        torch.save({"network": self.network.state_dict(), "num_timesteps": self.num_timesteps}, temporary)
        temporary.replace(path)

    def load(self, path: Path) -> None:
        saved = torch.load(path, map_location=self.network.device)
        self.network.load_state_dict(saved["network"])
        self.num_timesteps = int(saved["num_timesteps"])

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": self.network.checkpoint,
            "actions": self.network.question.options,
            "device": str(self.network.device),
            "parameters": sum(parameter.numel() for parameter in self.network.parameters()),
            "num_timesteps": self.num_timesteps,
            "exploration": round(self.exploration, 3),
            **self.last_update,
        }
