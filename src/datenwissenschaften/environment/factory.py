from speedlab_engine.console import Console

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.route.paths import route_path
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.settings import RetroSpeedlabConfig


def make_environment(
    wrapper_cls: type[StateMachineGymWrapper[RamInfo]], config: RetroSpeedlabConfig
) -> StateMachineGymWrapper[RamInfo]:
    console = Console(config.paths.collection, config.training.cartridge)
    listed = config.speedrun.game.version
    if console.version != listed:
        raise RuntimeError(
            f"{config.training.cartridge} is version {console.version}, but the collection lists {listed}"
        )
    return wrapper_cls(console, RouteMap(route_path(config)))
