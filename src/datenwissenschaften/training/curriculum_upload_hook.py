import httpx
from loguru import logger

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition

TIMEOUT_SECONDS = 30
UPLOADED_FIELDS = ("wins", "win_target", "mastered", "active", "has_checkpoint")


class CurriculumUploadHook:
    def __init__(self, context: RunContext, curriculum: CurriculumRun) -> None:
        self.context = context
        self.curriculum = curriculum
        self.settings = context.config.upload
        self.uploaded: dict | None = None

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        if self.settings.api_key is None:
            return
        states = [
            {"name": name, **{field: progress[field] for field in UPLOADED_FIELDS}}
            for name, progress in self.curriculum.curriculum.progress().items()
        ]
        payload = {
            "states": states,
            "levels": {level: list(members) for level, members in self.curriculum.targets.levels.items()},
            "level_times": self.curriculum.clock.times(),
        }
        if payload == self.uploaded:
            return
        try:
            response = httpx.put(
                f"{self.settings.url}/curricula/{self.context.game}",
                json=payload,
                headers={"X-API-Key": self.settings.api_key},
                timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            logger.error(f"Curriculum upload failed: {error}")
            return
        self.uploaded = payload
