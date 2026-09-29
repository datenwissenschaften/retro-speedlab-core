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
    curriculum_root = config.paths.cache_dir / "automatic_savestates" / training.game_identity / savestate
    state_names = tuple(state_cls.__name__ for state_cls in wrapper_cls.state_classes)
    curriculum = CurriculumRun(curriculum_root, state_names)
    return wrapper_cls(env, curriculum, Landmarks(curriculum_root / LANDMARKS_FILE), savestate)
