from datetime import UTC, datetime
from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.roms import import_roms
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.settings import RetroSpeedlabConfig

POWER_ON = "PowerOn"
SESSION_STAMP = "%Y%m%dT%H%M%S"
ROUTE_FILE = "route.json"


def make_environment(
    wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig, worker: int
) -> StateMachineGymWrapper:
    import_roms(config.paths.roms_path, config.paths.integrations_dir)
    env = retro.make(
        config.training.game,
        retro.State.NONE,
        render_mode="rgb_array",
        use_restricted_actions=retro.Actions.ALL,
    )
    env.initial_state = env.em.get_state()
    env.statename = f"{POWER_ON}-{datetime.now(UTC):{SESSION_STAMP}}-{worker}"
    return wrapper_cls(env, RouteMap(lambda state: route_path(config, state)))


def route_path(config: RetroSpeedlabConfig, state: str) -> Path:
    return config.paths.models_dir / config.training.game / state / ROUTE_FILE
