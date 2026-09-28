from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.training.context import RunContext


class StateModels:
    def __init__(self, agent: LayaAgent, context: RunContext, state_names: tuple[str, ...]) -> None:
        self.agent = agent
        self.context = context
        self.rollouts = {state_name: Rollout() for state_name in state_names}
        self.active: str | None = None

    @property
    def rollout(self) -> Rollout:
        return self.rollouts[self.require_active()]

    def require_active(self) -> str:
        if self.active is None:
            raise RuntimeError("No state model is active.")
        return self.active

    def activate(self, state_name: str) -> None:
        if state_name == self.active:
            return
        path = self.context.model_path(state_name)
        if path.is_file():
            self.agent.load(path)
        elif self.active is not None:
            self.agent.restart()
        self.active = state_name
        logger.debug(f"Laya model for {state_name} active at {self.agent.num_timesteps:,} trained decisions")

    def learn(self) -> None:
        self.agent.learn(self.rollout)
        self.rollouts[self.require_active()] = Rollout()

    def save(self) -> None:
        path = self.context.model_path(self.require_active())
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
