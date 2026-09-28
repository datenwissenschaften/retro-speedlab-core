from pathlib import Path

from fakes import write_config

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.settings import load_config
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.state_models import StateModels


class SwappingAgent:
    def __init__(self) -> None:
        self.weights = "pretrained"
        self.num_timesteps = 0
        self.learned: list[int] = []

    def load(self, path: Path) -> None:
        self.weights = path.read_text()

    def save(self, path: Path) -> None:
        path.write_text(self.weights)

    def restart(self) -> None:
        self.weights = "pretrained"

    def learn(self, rollout) -> None:
        self.learned.append(len(rollout))


def test_every_state_trains_and_keeps_its_own_model(tmp_path: Path):
    agent = SwappingAgent()
    models = StateModels(agent, RunContext(load_config(write_config(tmp_path))), ("Survive", "Boss"))

    models.activate("Survive")
    agent.weights = "survivor"
    models.rollout.add("{}", "q", Decision(0, {"left": 1.0}, 1.0), 1.0, False)
    models.save()
    models.activate("Boss")
    boss_weights = agent.weights
    models.learn()
    models.activate("Survive")

    assert boss_weights == "pretrained"
    assert agent.learned == [0]
    assert agent.weights == "survivor"
    assert len(models.rollout) == 1
