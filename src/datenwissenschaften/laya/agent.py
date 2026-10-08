import random
from typing import Any

import torch

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.heads import Heads
from datenwissenschaften.laya.imitation import DemonstrationStep, imitation_loss
from datenwissenschaften.laya.network import Features, LayaNetwork
from datenwissenschaften.laya.policy import Policy
from datenwissenschaften.ppo import PpoLearner
from datenwissenschaften.rollout import Rollout

LEARNING_RATE = 3e-4
MINIBATCH = 64
NO_ADVICE = -1

Observation = dict[str, str]


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
    def decide(self, features: Features, exploration: float, advice: list[int | None]) -> list[Decision]:
        options, pooled = features
        probabilities = torch.softmax(self.policy.heads.policy(options), -1)
        behavior = (1 - exploration) * probabilities + exploration / probabilities.size(-1)
        actions = torch.distributions.Categorical(probs=behavior).sample().tolist()
        values = self.policy.heads.value(options, pooled).tolist()
        names = self.network.question.options
        return [
            Decision(
                action,
                dict(zip(names, probabilities[row].tolist(), strict=True)),
                float(behavior[row, action]),
                float(probabilities[row, action]),
                values[row],
                options[row].cpu(),
                pooled[row].cpu(),
                advice[row],
            )
            for row, action in enumerate(actions)
        ]

    def act(self, observation: Observation, exploration: float, advice: int | None) -> Decision:
        return self.decide(self.read([observation]), exploration, [advice])[0]

    def learn(self, rollout: Rollout[Decision], demonstrations: list[DemonstrationStep]) -> None:
        device = self.network.device
        options = torch.stack([decision.options for decision in rollout.decisions]).to(device)
        pooled = torch.stack([decision.pooled for decision in rollout.decisions]).to(device)
        advice = torch.tensor(
            [NO_ADVICE if decision.advice is None else decision.advice for decision in rollout.decisions],
            device=device,
        )
        demonstration = self._demonstration(demonstrations)
        heads = self.policy.heads

        def evaluate(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.log_softmax(heads.policy(options[index]), -1), heads.value(options[index], pooled[index])

        def imitate(index: torch.Tensor) -> torch.Tensor:
            advised = index[advice[index] != NO_ADVICE]
            loss = options.new_zeros(())
            if len(advised):
                loss = loss + imitation_loss(heads.policy, options[advised], advice[advised])
            if demonstration is not None:
                loss = loss + imitation_loss(heads.policy, *demonstration)
            return loss

        self.policy.last_update = {
            **self.policy.learner.update(rollout, evaluate, imitate),
            **self._agreement(rollout),
            "demonstration_decisions": 0 if demonstration is None else len(demonstration[1]),
        }
        self.policy.num_timesteps += len(rollout)

    def _agreement(self, rollout: Rollout[Decision]) -> dict[str, float]:
        names = list(self.network.question.options)
        advised = [decision for decision in rollout.decisions if decision.advice is not None]
        if not advised:
            return {}
        agreeing = sum(max(d.probabilities, key=d.probabilities.__getitem__) == names[d.advice] for d in advised)
        return {"advisor_agreement": agreeing / len(advised)}

    def _demonstration(self, demonstrations: list[DemonstrationStep]) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not demonstrations:
            return None
        sample = random.sample(demonstrations, min(len(demonstrations), MINIBATCH))
        options, _ = self.network.features([step.state for step in sample], [step.question for step in sample])
        return options, torch.tensor([step.action for step in sample], device=options.device)

    def restart(self) -> None:
        heads = Heads(self.network.scorer, self.network.width, len(self.network.question.options))
        heads = heads.to(self.network.device).eval()
        self.policy = Policy(heads, PpoLearner((heads.policy, heads.value), LEARNING_RATE, MINIBATCH), 0, {})

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
