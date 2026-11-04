from pathlib import Path
from typing import Any

import flyte
from flyte import TaskTemplate


def launch(task: TaskTemplate[..., None, Any], wrapper: str, config: Path) -> None:
    flyte.init()
    flyte.run(task, wrapper=wrapper, config=str(config))
