import re
from importlib.metadata import PackageNotFoundError, version

from loguru import logger

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.story_book import story_key
from datenwissenschaften.ui.control import ModelResetRequest, perform_model_reset

PACKAGE_NAME = "datenwissenschaften"
DEVELOPMENT_VERSION = "DEVELOPMENT"
VERSION_KEY = "engine-version"
FINGERPRINT_KEY = "database-fingerprint"
MODEL_LAYOUT_KEY = "model-layout"
MODEL_LAYOUT = "laya-per-state-shuffled-options"
MAJOR_MINOR = re.compile(r"^v?(\d+)\.(\d+)")


def engine_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return DEVELOPMENT_VERSION


class TrainingIdentity:
    def __init__(self, context: RunContext, database: JsonDatabase) -> None:
        self.context = context
        self.database = database
        identity = context.config.training.game_identity
        self.version_key = f"{VERSION_KEY}:{identity}"
        self.fingerprint_key = f"{FINGERPRINT_KEY}:{identity}"
        self.model_layout_key = f"{MODEL_LAYOUT_KEY}:{identity}"

    def require_compatible(self, env: StateMachineGymWrapper) -> None:
        current_version = engine_version()
        previous_version = self._stored(self.version_key)
        fingerprint = self.context.config.training.fingerprint
        if (
            self._release(previous_version) != self._release(current_version)
            or self._stored(self.fingerprint_key) != fingerprint
            or self._stored(self.model_layout_key) != MODEL_LAYOUT
        ):
            logger.warning(f"Training identity changed for {self.context.game}; starting fresh.")
            perform_model_reset(self.reset_request(env))
        self.database.set(self.version_key, current_version)
        self.database.set(self.fingerprint_key, fingerprint)
        self.database.set(self.model_layout_key, MODEL_LAYOUT)

    def _stored(self, key: str) -> str | None:
        return self.database.get(key) if self.database.contains(key) else None

    def reset_request(self, env: StateMachineGymWrapper) -> ModelResetRequest:
        paths = self.context.config.paths
        return ModelResetRequest(
            game=self.context.game,
            model_dir=self.context.model_dir,
            artifact_dirs=(paths.models_dir, paths.record_dir, paths.cache_dir),
            on_reset=lambda: self._forget(env),
        )

    def _forget(self, env: StateMachineGymWrapper) -> None:
        env.reset_training_memory()
        self.database.delete(story_key(self.context.config.training.game_identity))

    @staticmethod
    def _release(engine: str | None) -> tuple[str, ...] | None:
        match = MAJOR_MINOR.match(engine) if engine is not None else None
        return None if match is None else match.groups()
