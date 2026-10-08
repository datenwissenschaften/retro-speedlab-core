from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent, Observation
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.laya.policy import Policy
from datenwissenschaften.rollout import Rollout
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.model_store import ModelStore


class StateModels:
    def __init__(self, agent: LayaAgent, context: RunContext, state_names: tuple[str, ...]) -> None:
        self.agent = agent
        self.context = context
        self.state_names = state_names
        self.policies: dict[str, Policy] = {}
        self.rollouts: dict[tuple[str, int], Rollout[Decision]] = {}
        self.store = ModelStore()
        self.active: str | None = None

    def require_active(self) -> str:
        if self.active is None:
            raise RuntimeError("No state model is active.")
        return self.active

    def activate(self, state_name: str) -> None:
        if state_name == self.active:
            return
        if state_name not in self.policies:
            self.agent.restart()
            checkpoint = self.store.read(self.context.model_path(state_name)).result()
            if checkpoint is not None:
                self.agent.policy.restore(checkpoint)
            self.policies[state_name] = self.agent.policy
            logger.debug(f"Laya model for {state_name} loaded at {self.agent.num_timesteps:,} trained decisions")
        self.agent.policy = self.policies[state_name]
        self.active = state_name

    def decide(
        self,
        observations: list[Observation],
        states: list[str],
        exploration: dict[str, float],
        advice: list[int | None],
    ) -> list[Decision]:
        options, pooled = self.agent.read(observations)
        decisions: dict[int, Decision] = {}
        for state_name in dict.fromkeys(states):
            rows = [row for row, state in enumerate(states) if state == state_name]
            self.activate(state_name)
            chosen = self.agent.decide(
                (options[rows], pooled[rows]), exploration[state_name], [advice[row] for row in rows]
            )
            decisions.update(zip(rows, chosen, strict=True))
        return [decisions[row] for row in range(len(states))]

    def rollout(self, state_name: str, environment: int) -> Rollout[Decision]:
        return self.rollouts.setdefault((state_name, environment), Rollout())

    def collected(self, state_name: str) -> int:
        return sum(len(rollout) for (state, _), rollout in self.rollouts.items() if state == state_name)

    def timesteps(self, state_name: str) -> int:
        self.activate(state_name)
        return self.agent.num_timesteps + self.collected(state_name)

    def learn(self, state_name: str, demonstrations: list[DemonstrationStep]) -> None:
        self.activate(state_name)
        keys = [key for key in self.rollouts if key[0] == state_name]
        self.agent.learn(Rollout.joined([self.rollouts.pop(key) for key in keys]), demonstrations)

    def save(self) -> None:
        self.store.write(self.context.model_path(self.require_active()), self.agent.policy.checkpoint)

    def close(self) -> None:
        self.store.close()
