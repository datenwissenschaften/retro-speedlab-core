from importlib import import_module
from pathlib import Path

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.training.trainer import LayaTrainer
from datenwissenschaften.workflow.environment import speedlab


def wrapper_class(reference: str) -> type[StateMachineGymWrapper]:
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError(
            f"A wrapper is named as 'module:Class', for example 'src.game.wrapper:GameWrapper': {reference}"
        )
    return getattr(import_module(module), name)


@speedlab.task
def train_session(wrapper: str, config: str) -> None:
    LayaTrainer(wrapper_class(wrapper), Path(config)).train_session()


@speedlab.task
def training(wrapper: str, config: str) -> None:
    while True:
        train_session(wrapper, config)
