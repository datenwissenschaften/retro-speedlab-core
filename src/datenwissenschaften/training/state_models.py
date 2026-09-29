from concurrent.futures import Future

from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.model_store import Checkpoint, ModelStore


class StateModels:
    def __init__(self, agent: LayaAgent, context: RunContext, state_names: tuple[str, ...]) -> None:
        self.agent = agent
        self.context = context
        self.state_names = state_names
        self.rollouts = {state_name: Rollout() for state_name in state_names}
        self.store = ModelStore()
        self.active: str | None = None
        self.prefetched: tuple[str, Future[Checkpoint | None]] | None = None

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
        checkpoint = self._fetch(state_name).result()
        if checkpoint is not None:
            self.agent.restore(checkpoint)
        elif self.active is not None:
            self.agent.restart()
        self.active = state_name
        self._prefetch_following(state_name)
        logger.debug(f"Laya model for {state_name} active at {self.agent.num_timesteps:,} trained decisions")

    def learn(self) -> None:
        self.agent.learn(self.rollout)
        self.rollouts[self.require_active()] = Rollout()

    def save(self) -> None:
        self.store.write(self.context.model_path(self.require_active()), self.agent.checkpoint)

    def close(self) -> None:
        self.store.close()

    def _fetch(self, state_name: str) -> Future[Checkpoint | None]:
        prefetched, self.prefetched = self.prefetched, None
        if prefetched is not None and prefetched[0] == state_name:
            return prefetched[1]
        return self.store.read(self.context.model_path(state_name))

    def _prefetch_following(self, state_name: str) -> None:
        following = self.state_names[(self.state_names.index(state_name) + 1) % len(self.state_names)]
        if following != state_name:
            self.prefetched = following, self.store.read(self.context.model_path(following))
