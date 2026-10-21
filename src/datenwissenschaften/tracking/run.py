import os
import time

from loguru import logger
from mlflow import MlflowClient, flush_async_logging
from mlflow.entities import Metric, Param, RunStatus
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import TrackingSettings

TOKEN_VARIABLE = "MLFLOW_TRACKING_TOKEN"
MILLISECONDS_PER_SECOND = 1000


class TrackingRun:
    def __init__(self, client: MlflowClient, run_id: str) -> None:
        self.client = client
        self.run_id = run_id

    @classmethod
    def start(cls, client: MlflowClient, experiment: str, name: str, parameters: dict[str, str]) -> "TrackingRun":
        found = client.get_experiment_by_name(experiment)
        experiment_id = client.create_experiment(experiment) if found is None else found.experiment_id
        run_id = client.create_run(experiment_id, run_name=name).info.run_id
        client.log_batch(run_id, params=[Param(key, value) for key, value in parameters.items()])
        return cls(client, run_id)

    def log(self, metrics: dict[str, float], step: int) -> None:
        timestamp = int(time.time() * MILLISECONDS_PER_SECOND)
        entries = [Metric(key, value, timestamp, step) for key, value in metrics.items()]
        self.client.log_batch(self.run_id, metrics=entries, synchronous=False)

    def finish(self) -> None:
        flush_async_logging()
        self.client.set_terminated(self.run_id, RunStatus.to_string(RunStatus.FINISHED))


def start_tracking(
    settings: TrackingSettings, experiment: str, name: str, parameters: dict[str, str]
) -> TrackingRun | None:
    if settings.token is None:
        return None
    os.environ[TOKEN_VARIABLE] = settings.token
    try:
        return TrackingRun.start(MlflowClient(tracking_uri=settings.url), experiment, name, parameters)
    except MlflowException as error:
        logger.warning(f"MLflow at {settings.url} is unreachable, this session is not tracked: {error}")
        return None
