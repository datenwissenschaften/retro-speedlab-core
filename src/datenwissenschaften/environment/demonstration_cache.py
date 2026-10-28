import hashlib
import pickle
import sys
from pathlib import Path

from loguru import logger

from datenwissenschaften.curriculum import atomic_write
from datenwissenschaften.environment.demonstration import Demonstrations, StartPoints, load_demonstrations
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.settings import RetroSpeedlabPaths

CACHE_DIR = "demonstrations"
CACHE_SUFFIX = ".pickle"
ENGINE_DIR = Path(__file__).parents[1]


def game_dir(wrapper_cls: type[StateMachineGymWrapper]) -> Path:
    top = sys.modules[wrapper_cls.__module__.split(".")[0]]
    if hasattr(top, "__path__"):
        return Path(next(iter(top.__path__)))
    return Path(top.__file__).parent


def source_files(wrapper_cls: type[StateMachineGymWrapper], paths: RetroSpeedlabPaths) -> list[Path]:
    integrations = [path for path in paths.integrations_dir.rglob("*") if path.is_file()]
    code = [*game_dir(wrapper_cls).rglob("*.py"), *ENGINE_DIR.rglob("*.py")]
    return sorted([*paths.demonstrations_dir.glob("*.bk2"), *integrations, *code])


def fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_demonstrations(
    wrapper: StateMachineGymWrapper, paths: RetroSpeedlabPaths
) -> tuple[Demonstrations, StartPoints]:
    cache = paths.cache_dir / CACHE_DIR / f"{fingerprint(source_files(type(wrapper), paths))}{CACHE_SUFFIX}"
    if cache.is_file():
        demonstrations, starts = pickle.loads(cache.read_bytes())
        logger.info(f"Reused the replayed demonstrations of {len(demonstrations)} states from {cache.name}")
        return demonstrations, starts
    replayed = load_demonstrations(wrapper, paths.demonstrations_dir)
    for stale in cache.parent.glob(f"*{CACHE_SUFFIX}"):
        stale.unlink()
    atomic_write(cache, pickle.dumps(replayed))
    return replayed
