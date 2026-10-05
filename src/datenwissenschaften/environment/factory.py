from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.roms import import_roms
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.landmarks import Landmarks

LANDMARKS_FILE = "landmarks.json"
POWER_ON = "PowerOn"


def make_environment(wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig) -> StateMachineGymWrapper:
    training = config.training
    import_roms(config.paths.roms_path, config.paths.integrations_dir)
    record_dir = config.paths.record_dir / training.game
    record_dir.mkdir(parents=True, exist_ok=True)
    env = retro.make(
        training.game,
        retro.State.NONE,
        render_mode="rgb_array",
        record=str(record_dir),
        use_restricted_actions=retro.Actions.ALL,
    )
    emulator = env.unwrapped
    emulator.initial_state = emulator.em.get_state()
    emulator.statename = POWER_ON
    root = curriculum_root(config)
    curriculum = CurriculumRun(root, state_names(wrapper_cls), POWER_ON, config.paths.curriculum_dir)
    return wrapper_cls(env, curriculum, Landmarks(root / LANDMARKS_FILE), POWER_ON)


def curriculum_root(config: RetroSpeedlabConfig) -> Path:
    return config.paths.cache_dir / "curriculum" / config.training.game_identity


def state_names(wrapper_cls: type[StateMachineGymWrapper]) -> tuple[str, ...]:
    return tuple(state_cls.__name__ for state_cls in wrapper_cls.state_classes)
