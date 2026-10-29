from pathlib import Path

import flyte
from flyte import TaskTemplate


def launch(task: TaskTemplate, wrapper: str, config: Path) -> None:
    flyte.init()
    flyte.run(task, wrapper=wrapper, config=str(config))
