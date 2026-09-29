from dataclasses import dataclass
from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

MODEL_FILENAME = "laya.pt"


@dataclass(slots=True, frozen=True)
class RunContext:
    config: RetroSpeedlabConfig
    savestate: str

    @property
    def game(self) -> str:
        return self.config.training.game

    @property
    def record_root(self) -> Path:
        return self.config.paths.record_dir

    @property
    def record_dir(self) -> Path:
        return self.record_root / self.game / self.savestate

    @property
    def model_dir(self) -> Path:
        return self.config.paths.models_dir / self.config.training.game_identity / self.savestate

    def model_path(self, state_name: str) -> Path:
        return self.model_dir / state_name / MODEL_FILENAME
