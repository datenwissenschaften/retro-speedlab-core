import copy
import io
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from datenwissenschaften.advisor.advice import Advice, AdvisorDecision
from datenwissenschaften.advisor.network import ActorCritic
from datenwissenschaften.laya.imitation import IMITATION_WEIGHT, DemonstrationStep
from datenwissenschaften.ppo import PpoLearner
from datenwissenschaften.rollout import Rollout
from datenwissenschaften.training.model_store import ModelStore

LEARNING_RATE = 3e-4
MINIBATCH = 256


@dataclass(slots=True)
class StateAdvisor:
    learning: ActorCritic
    acting: ActorCritic
    learner: PpoLearner
    num_timesteps: int

    def checkpoint(self) -> io.BytesIO:
        buffer = io.BytesIO()
        state = {"model": self.learning.state_dict(), "learner": self.learner.state_dict()}
        torch.save({**state, "num_timesteps": self.num_timesteps}, buffer)
        return buffer


class Advisors:
    def __init__(self, actions: tuple[str, ...], device: str, path: Callable[[str], Path]) -> None:
        self.actions = actions
        self.device = torch.device(device)
        self.path = path
        self.models: dict[str, StateAdvisor] = {}
        self.lock = threading.Lock()
        self.store = ModelStore()

    @torch.no_grad()
    def act(self, states: list[str], inputs: list[np.ndarray]) -> list[AdvisorDecision]:
        decisions: dict[int, AdvisorDecision] = {}
        for state in dict.fromkeys(states):
            rows = [row for row, name in enumerate(states) if name == state]
            batch = np.stack([inputs[row] for row in rows])
            with self.lock:
                log_probs, values = self._model(state, batch.shape[1]).acting(torch.from_numpy(batch).to(self.device))
            probabilities = log_probs.exp()
            actions = torch.multinomial(probabilities, 1).squeeze(1).tolist()
            chosen = probabilities[torch.arange(len(rows)), actions].tolist()
            for row, action, probability, value in zip(rows, actions, chosen, values.tolist(), strict=True):
                decisions[row] = AdvisorDecision(action, probability, value, inputs[row])
        return [decisions[row] for row in range(len(states))]

    @torch.no_grad()
    def advise(self, state: str, inputs: np.ndarray) -> Advice:
        with self.lock:
            log_probs, _ = self._model(state, len(inputs)).acting(torch.from_numpy(inputs)[None].to(self.device))
        probabilities = log_probs[0].exp().tolist()
        return Advice(int(np.argmax(probabilities)), dict(zip(self.actions, probabilities, strict=True)))

    def learn(
        self, state: str, rollout: Rollout[AdvisorDecision], demonstrations: list[DemonstrationStep]
    ) -> dict[str, float]:
        advisor = self.models[state]
        inputs = torch.from_numpy(np.stack([decision.inputs for decision in rollout.decisions])).to(self.device)
        sample = random.sample(demonstrations, min(len(demonstrations), MINIBATCH))
        shown = torch.from_numpy(np.stack([step.inputs for step in sample])).to(self.device) if sample else None
        moves = torch.tensor([step.action for step in sample], device=self.device)

        def evaluate(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return advisor.learning(inputs[index])

        def imitate(index: torch.Tensor) -> torch.Tensor:
            if shown is None:
                return inputs.new_zeros(())
            log_probs, _ = advisor.learning(shown)
            return -IMITATION_WEIGHT * log_probs.gather(1, moves[:, None]).mean()

        metrics = advisor.learner.update(rollout, evaluate, imitate)
        advisor.learning.normalizer.update(inputs)
        advisor.num_timesteps += len(rollout)
        with self.lock:
            advisor.acting.load_state_dict(advisor.learning.state_dict())
        self.store.write(self.path(state), advisor.checkpoint)
        return {**metrics, "num_timesteps": advisor.num_timesteps}

    def close(self) -> None:
        self.store.close()

    def _model(self, state: str, inputs: int) -> StateAdvisor:
        if state in self.models:
            return self.models[state]
        learning = ActorCritic(inputs, len(self.actions)).to(self.device)
        learner = PpoLearner((learning.policy, learning.value), LEARNING_RATE, MINIBATCH)
        advisor = StateAdvisor(learning, copy.deepcopy(learning).eval(), learner, 0)
        checkpoint = self.store.read(self.path(state)).result()
        if checkpoint is not None:
            learning.load_state_dict(checkpoint["model"])
            learner.load_state_dict(checkpoint["learner"])
            advisor.acting.load_state_dict(checkpoint["model"])
            advisor.num_timesteps = int(checkpoint["num_timesteps"])
        self.models[state] = advisor
        return advisor
