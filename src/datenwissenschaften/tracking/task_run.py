from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from datenwissenschaften.settings import load_config
from datenwissenschaften.tracking.run import start_tracking

TASKS_SUFFIX = " tasks"


@contextmanager
def task_run(config_path: Path, task: str) -> Iterator[None]:
    config = load_config(config_path)
    experiment = f"{config.training.game_identity}{TASKS_SUFFIX}"
    run = start_tracking(config.tracking, experiment, task, {"release": config.ui.release})
    if run is None:
        yield
        return
    try:
        yield
    except BaseException as error:
        run.fail(error)
        raise
    run.finish()
