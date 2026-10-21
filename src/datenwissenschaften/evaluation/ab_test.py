from pathlib import Path

from loguru import logger

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.environment.factory import POWER_ON, make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.evaluation.player import Attempt, Player
from datenwissenschaften.evaluation.snapshot import with_models
from datenwissenschaften.evaluation.statistics import Comparison, compare
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tracking.run import start_tracking
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.session import MAIN_ENVIRONMENT
from datenwissenschaften.training.trainer import laya_network, phases, state_models

EPISODES = 5
DECISIONS = 600
EXPERIMENT_SUFFIX = " A/B"


def evaluation_starts(env: StateMachineGymWrapper) -> tuple[str | None, ...]:
    curriculum = env.curriculum.curriculum
    frontier = curriculum.active_state()
    if frontier is None or not curriculum.has_checkpoint(frontier):
        return (None,)
    return (None, frontier)


class ABTest:
    def __init__(self, wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig) -> None:
        self.wrapper_cls = wrapper_cls
        self.config = config

    def run(self, before_dir: Path) -> dict[str, Comparison]:
        env = make_environment(self.wrapper_cls, self.config, MAIN_ENVIRONMENT, False)
        try:
            starts = evaluation_starts(env)
            network = laya_network(self.wrapper_cls, self.config)
            before = self._attempts(env, network, before_dir, starts)
            after = self._attempts(env, network, self.config.paths.models_dir, starts)
        finally:
            env.close()
        comparisons = {
            "score": compare([attempt.score for attempt in before], [attempt.score for attempt in after]),
            "progress": compare([attempt.progress for attempt in before], [attempt.progress for attempt in after]),
        }
        self._report(comparisons, starts)
        return comparisons

    def _attempts(
        self, env: StateMachineGymWrapper, network: LayaNetwork, models_dir: Path, starts: tuple[str | None, ...]
    ) -> list[Attempt]:
        context = RunContext(with_models(self.config, models_dir), POWER_ON)
        models = state_models(self.wrapper_cls, network, context)
        actions = tuple(self.wrapper_cls.action_descriptions)
        advisors = Advisors(actions, configure_accelerator(), context.advisor_path)
        env.observer.advisor = advisors.advise
        player = Player(env, models, phases(self.wrapper_cls))
        try:
            return [player.play(start, DECISIONS) for start in starts for _ in range(EPISODES)]
        finally:
            advisors.close()
            models.close()

    def _report(self, comparisons: dict[str, Comparison], starts: tuple[str | None, ...]) -> None:
        for name, comparison in comparisons.items():
            logger.info(
                f"A/B {name}: {comparison.before:.2f} before, {comparison.after:.2f} after, "
                f"difference [{comparison.difference_low:.2f}, {comparison.difference_high:.2f}], "
                f"P(improvement) {comparison.probability_of_improvement:.2f}"
            )
        parameters = {
            "episodes": str(EPISODES),
            "decisions": str(DECISIONS),
            "starts": ", ".join(start or POWER_ON for start in starts),
        }
        experiment = f"{self.config.training.game_identity}{EXPERIMENT_SUFFIX}"
        run = start_tracking(self.config.tracking, experiment, self.config.ui.release, parameters)
        if run is None:
            return
        run.log({key: value for name, item in comparisons.items() for key, value in item.metrics(name).items()}, 0)
        run.finish()
