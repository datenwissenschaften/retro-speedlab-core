from collections.abc import Callable
from pathlib import Path

import typer
from loguru import logger

from datenwissenschaften.knowledge import KNOWLEDGE_FILE, load_knowledge
from datenwissenschaften.speedrun.hints import HINTS_FILE, load_hints
from datenwissenschaften.speedrun.paths import atomic_write

LOADERS: dict[str, Callable[[Path], object]] = {HINTS_FILE: load_hints, KNOWLEDGE_FILE: load_knowledge}


def save(draft: Path, target: Path) -> None:
    if target.name not in LOADERS:
        raise typer.BadParameter(f"{target.name} is not one of {', '.join(LOADERS)}")
    if not draft.is_file():
        raise typer.BadParameter(f"There is no draft at {draft}")
    LOADERS[target.name](draft)
    atomic_write(target, draft.read_bytes())
    logger.info(f"{draft} is valid and saved to {target}")


if __name__ == "__main__":
    typer.run(save)
