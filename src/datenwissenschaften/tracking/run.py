import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from loguru import logger
from mlflow import MlflowClient, flush_async_logging
from mlflow.entities import Metric, Param, RunStatus
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import TrackingSettings

TOKEN_VARIABLE = "MLFLOW_TRACKING_TOKEN"
MILLISECONDS_PER_SECOND = 1000
ERROR_TAG = "error"
MAX_TAG_LENGTH = 5000
STOPS = (KeyboardInterrupt, SystemExit)


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
        run = cls(client, run_id)
        ACTIVE_RUNS.add(run)
        return run

    def log(self, metrics: dict[str, float], step: int) -> None:
        timestamp = int(time.time() * MILLISECONDS_PER_SECOND)
        entries = [Metric(key, value, timestamp, step) for key, value in metrics.items()]
        self.client.log_batch(self.run_id, metrics=entries, synchronous=False)

    def end(self, error: BaseException | None) -> None:
        if error is None:
            self.finish()
        elif isinstance(error, STOPS):
            self.kill()
        else:
            self.fail(error)

    def finish(self) -> None:
        self._end(RunStatus.FINISHED)

    def kill(self) -> None:
        ACTIVE_RUNS.discard(self)
        end_time = int(time.time() * MILLISECONDS_PER_SECOND)
        self.client._tracking_client.store.update_run_info(self.run_id, RunStatus.KILLED, end_time, None)

    def fail(self, error: BaseException) -> None:
        self.client.set_tag(self.run_id, ERROR_TAG, f"{type(error).__name__}: {error}"[:MAX_TAG_LENGTH])
        self._end(RunStatus.FAILED)

    def _end(self, status: int) -> None:
        ACTIVE_RUNS.discard(self)
        flush_async_logging()
        self.client.set_terminated(self.run_id, RunStatus.to_string(status))


ACTIVE_RUNS: set[TrackingRun] = set()


def kill_active_runs() -> None:
    for run in tuple(ACTIVE_RUNS):
        run.kill()


@contextmanager
def ending(run: TrackingRun | None) -> Iterator[None]:
    if run is None:
        yield
        return
    try:
        yield
    except BaseException as error:
        run.end(error)
        raise
    run.end(None)


def tracking_client(settings: TrackingSettings) -> MlflowClient | None:
    if settings.token is None:
        return None
    os.environ[TOKEN_VARIABLE] = settings.token
    return MlflowClient(tracking_uri=settings.url)


def start_tracking(
    settings: TrackingSettings, experiment: str, name: str, parameters: dict[str, str]
) -> TrackingRun | None:
    client = tracking_client(settings)
    if client is None:
        return None
    try:
        return TrackingRun.start(client, experiment, name, parameters)
    except MlflowException as error:
        logger.warning(f"MLflow at {settings.url} is unreachable, this run is not tracked: {error}")
        return None
