from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.roms import import_roms
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.landmarks import Landmarks

LANDMARKS_FILE = "landmarks.json"
FULL_GAME = "FullGame"


def make_environment(wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig) -> StateMachineGymWrapper:
    training = config.training
    import_roms(config.paths.roms_path)
    record_dir = config.paths.record_dir / training.game
    record_dir.mkdir(parents=True, exist_ok=True)
    env = retro.make(
        training.game,
        retro.State.NONE,
        render_mode="rgb_array",
        record=str(record_dir),
        use_restricted_actions=retro.Actions.ALL,
    )
    root = curriculum_root(config)
    curriculum = CurriculumRun(root, state_names(wrapper_cls), FULL_GAME, config.paths.curriculum_dir)
    return wrapper_cls(env, curriculum, Landmarks(root / LANDMARKS_FILE), FULL_GAME)


def curriculum_root(config: RetroSpeedlabConfig) -> Path:
    return config.paths.cache_dir / "curriculum" / config.training.game_identity


def state_names(wrapper_cls: type[StateMachineGymWrapper]) -> tuple[str, ...]:
    return tuple(state_cls.__name__ for state_cls in wrapper_cls.state_classes)
