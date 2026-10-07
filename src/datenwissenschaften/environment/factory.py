from datetime import UTC, datetime
from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.level_clock import LevelClock
from datenwissenschaften.environment.levels import LevelTargets, curriculum_targets, level_map
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.roms import import_roms
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.landmarks import Landmarks

LANDMARKS_FILE = "landmarks.json"
LEVEL_TIMES_FILE = "level_times.json"
POWER_ON = "PowerOn"
SESSION_STAMP = "%Y%m%dT%H%M%S"


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
    emulator.statename = f"{POWER_ON}-{datetime.now(UTC):{SESSION_STAMP}}"
    root = curriculum_root(config)
    levels = level_map(wrapper_cls.levels, wrapper_cls.state_classes)
    targets = curriculum_targets(wrapper_cls.state_classes, levels)
    clock = LevelClock(root / LEVEL_TIMES_FILE, emulator.em.get_screen_rate())
    curriculum = CurriculumRun(root, LevelTargets(targets, levels), POWER_ON, config.paths.curriculum_dir, clock)
    return wrapper_cls(env, curriculum, Landmarks(root / LANDMARKS_FILE), POWER_ON)


def curriculum_root(config: RetroSpeedlabConfig) -> Path:
    return config.paths.cache_dir / "curriculum" / config.training.game_identity
