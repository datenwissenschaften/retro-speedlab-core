from pathlib import Path

from datenwissenschaften.settings import load_config
from datenwissenschaften.tracking.stale import close_stale_training_runs
from datenwissenschaften.tracking.task_run import task_run
from datenwissenschaften.training.trainer import LayaTrainer
from datenwissenschaften.workflow.environment import speedlab
from datenwissenschaften.workflow.evaluation import ab_test, snapshot
from datenwissenschaften.workflow.wrapper import wrapper_class


@speedlab.task
def train_session(wrapper: str, config: str) -> None:
    close_stale_training_runs(load_config(Path(config)))
    with task_run(Path(config), "train_session"):
        LayaTrainer(wrapper_class(wrapper), Path(config)).train_session()


@speedlab.task
def training(wrapper: str, config: str) -> None:
    while True:
        before = snapshot(config)
        train_session(wrapper, config)
        ab_test(wrapper, config, before)
