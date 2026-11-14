from importlib import import_module

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo


def wrapper_class(reference: str) -> type[StateMachineGymWrapper[RamInfo]]:
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError(
            f"A wrapper is named as 'module:Class', for example 'speedrun.comet_courier.wrapper:Wrapper': {reference}"
        )
    return getattr(import_module(module), name)
