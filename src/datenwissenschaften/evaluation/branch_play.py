from dataclasses import replace
from pathlib import Path

import msgspec
import typer

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.environment.factory import POWER_ON, make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.evaluation.player import Attempt, Player
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.session import MAIN_ENVIRONMENT
from datenwissenschaften.training.trainer import laya_network, phases, state_models
from datenwissenschaften.workflow.wrapper import wrapper_class

EPISODES = 5
DECISIONS = 3000


def branch_config(config: Path, branch: Path) -> RetroSpeedlabConfig:
    settings = load_config(config)
    return replace(settings, paths=replace(settings.paths, integrations_dir=branch / "integrations"))


def play_attempts(wrapper_cls: type[StateMachineGymWrapper], settings: RetroSpeedlabConfig) -> list[Attempt]:
    env = make_environment(wrapper_cls, settings, MAIN_ENVIRONMENT, False)
    context = RunContext(settings, POWER_ON)
    models = state_models(wrapper_cls, laya_network(wrapper_cls, settings), context)
    advisors = Advisors(tuple(wrapper_cls.action_descriptions), configure_accelerator(), context.advisor_path)
    env.observer.advisor = advisors.advise
    player = Player(env, models, phases(wrapper_cls))
    try:
        return [player.play(DECISIONS) for _ in range(EPISODES)]
    finally:
        advisors.close()
        models.close()
        env.close()


def play_branch(wrapper: str, config: Path, branch: Path, output: Path) -> None:
    attempts = play_attempts(wrapper_class(wrapper), branch_config(config, branch))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(msgspec.json.encode(attempts))


if __name__ == "__main__":
    typer.run(play_branch)
