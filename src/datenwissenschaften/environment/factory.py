from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.roms import import_roms, import_savestates
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.landmarks import Landmarks

LANDMARKS_FILE = "landmarks.json"


def make_environment(
    wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig, savestate: str
) -> StateMachineGymWrapper:
    training = config.training
    import_roms(config.paths.roms_path)
    import_savestates(training.game, config.paths.savestates_dir)
    record_dir = config.paths.record_dir / training.game / savestate
    record_dir.mkdir(parents=True, exist_ok=True)
    env = retro.make(training.game, savestate, render_mode="rgb_array", record=str(record_dir))
    root = curriculum_root(config, savestate)
    curriculum = CurriculumRun(root, state_names(wrapper_cls), savestate)
    return wrapper_cls(env, curriculum, Landmarks(root / LANDMARKS_FILE), savestate)


def curriculum_root(config: RetroSpeedlabConfig, savestate: str) -> Path:
    return config.paths.cache_dir / "automatic_savestates" / config.training.game_identity / savestate


def state_names(wrapper_cls: type[StateMachineGymWrapper]) -> tuple[str, ...]:
    return tuple(state_cls.__name__ for state_cls in wrapper_cls.state_classes)
