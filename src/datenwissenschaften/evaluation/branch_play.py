from dataclasses import dataclass, replace
from pathlib import Path

import msgspec
import typer

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.environment.factory import POWER_ON, make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper, phases
from datenwissenschaften.evaluation.player import Attempt, Player
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.session import MAIN_ENVIRONMENT
from datenwissenschaften.training.trainer import laya_network, state_models
from datenwissenschaften.workflow.wrapper import wrapper_class

EPISODES = 5
DECISIONS = 3000


@dataclass(slots=True, frozen=True)
class BranchPlay:
    start: str
    attempts: list[Attempt]


def branch_config(config: Path, branch: Path) -> RetroSpeedlabConfig:
    settings = load_config(config)
    paths = replace(settings.paths, integrations_dir=branch / "integrations", curriculum_dir=branch / "curriculum")
    return replace(settings, paths=paths)


def duel_start(env: StateMachineGymWrapper) -> str | None:
    curriculum = env.curriculum.curriculum
    frontier = curriculum.active_state()
    return None if frontier is None else curriculum.episode_start_state(frontier)


def play_attempts(wrapper_cls: type[StateMachineGymWrapper], settings: RetroSpeedlabConfig) -> BranchPlay:
    env = make_environment(wrapper_cls, settings, MAIN_ENVIRONMENT, False)
    start = duel_start(env)
    context = RunContext(settings, POWER_ON)
    models = state_models(wrapper_cls, laya_network(wrapper_cls, settings), context)
    advisors = Advisors(tuple(wrapper_cls.action_descriptions), configure_accelerator(), context.advisor_path)
    env.observer.advisor = advisors.advise
    player = Player(env, models, phases(wrapper_cls))
    try:
        return BranchPlay(start or POWER_ON, [player.play(start, DECISIONS) for _ in range(EPISODES)])
    finally:
        advisors.close()
        models.close()
        env.close()


def play_branch(wrapper: str, config: Path, branch: Path, output: Path) -> None:
    played = play_attempts(wrapper_class(wrapper), branch_config(config, branch))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(msgspec.json.encode(played))


if __name__ == "__main__":
    typer.run(play_branch)
