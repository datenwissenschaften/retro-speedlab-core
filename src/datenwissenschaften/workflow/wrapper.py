from importlib import import_module

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper


def wrapper_class(reference: str) -> type[StateMachineGymWrapper]:
    module, separator, name = reference.partition(":")
    if not separator:
        raise ValueError(
            f"A wrapper is named as 'module:Class', for example 'src.game.wrapper:GameWrapper': {reference}"
        )
    return getattr(import_module(module), name)
