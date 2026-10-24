from loguru import logger
from mlflow import MlflowClient
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tracking.run import tracking_client
from datenwissenschaften.tracking.task_run import TASKS_SUFFIX

RUNNING = "attributes.status = 'RUNNING'"
TRAINING_TASKS = frozenset({"train_session"})
LAB_TASKS = frozenset({"lab_run"})


def _close(client: MlflowClient, experiment: str, names: frozenset[str] | None) -> None:
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return
    for run in client.search_runs([found.experiment_id], filter_string=RUNNING):
        if names is not None and run.info.run_name not in names:
            continue
        client.set_terminated(run.info.run_id, RunStatus.to_string(RunStatus.KILLED))
        logger.info(f"MLflow run {run.info.run_name} in {experiment} ended without closing; it is marked killed")


def _close_stale(config: RetroSpeedlabConfig, groups: tuple[tuple[str, frozenset[str] | None], ...]) -> None:
    client = tracking_client(config.tracking)
    if client is None:
        return
    try:
        for experiment, names in groups:
            _close(client, experiment, names)
    except MlflowException as error:
        logger.warning(f"MLflow at {config.tracking.url} is unreachable, stale runs stay open: {error}")


def close_stale_training_runs(config: RetroSpeedlabConfig) -> None:
    identity = config.training.game_identity
    _close_stale(config, ((identity, None), (f"{identity}{TASKS_SUFFIX}", TRAINING_TASKS)))


def close_stale_lab_runs(config: RetroSpeedlabConfig) -> None:
    _close_stale(config, ((f"{config.training.game_identity}{TASKS_SUFFIX}", LAB_TASKS),))
