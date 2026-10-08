import json
import subprocess
import time
from pathlib import Path

import httpx
from loguru import logger

from datenwissenschaften.environment.curriculum_run import FULL_RUN
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.serialization import to_json_value
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.system import system_metadata
from datenwissenschaften.training.video_render import count_frames, render_video

TIMEOUT_SECONDS = 120
FIRST_RETRY_SECONDS = 60.0
MAX_RETRY_SECONDS = 3600.0


class UploadHook:
    def __init__(self, context: RunContext, agent: LayaAgent, frame_rate: float, levels: frozenset[str]) -> None:
        self.context = context
        self.agent = agent
        self.frame_rate = frame_rate
        self.settings = context.config.upload
        self.levels = levels
        self.pending: list[tuple[EpisodeRecord, str]] = []
        self.failures = 0
        self.retry_at = 0.0

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        if episode.curriculum_state in self.levels and episode.curriculum_succeeded:
            self.pending.append((episode, episode.curriculum_state))
        elif episode.won and episode.started_from_initial_savestate:
            self.pending.append((episode, FULL_RUN))

    def on_update(self) -> None:
        if not self.pending or time.monotonic() < self.retry_at:
            return
        if self.settings.api_key is None:
            logger.info("Upload API key is not configured. Skipping episode upload.")
            self.pending.clear()
            return
        try:
            for beaten in list(self.pending):
                self._upload(*beaten, self.settings.api_key)
                self.pending.remove(beaten)
            self.failures = 0
        except (httpx.HTTPError, subprocess.CalledProcessError) as error:
            self.failures += 1
            delay = min(FIRST_RETRY_SECONDS * 2 ** (self.failures - 1), MAX_RETRY_SECONDS)
            self.retry_at = time.monotonic() + delay
            logger.error(f"Episode upload failed: {error}; retrying in {delay:.0f} s")

    def _details(self, episode: EpisodeRecord) -> str:
        details = {
            "laya": self.agent.metadata(),
            "system": system_metadata(),
            "episode": episode.episode_index,
            "steps": episode.step_count,
            "score": episode.score,
        }
        return json.dumps(to_json_value(details), sort_keys=True)

    def _upload(self, episode: EpisodeRecord, level: str, api_key: str) -> None:
        recording = Path(episode.bk2_path)
        video = render_video(self.context.config.paths, recording)
        with video.open("rb") as video_file:
            response = httpx.post(
                f"{self.settings.url}/runs",
                files={"video": (video.name, video_file, "video/mp4")},
                data={
                    "game": self.context.game,
                    "level": level,
                    "frames": str(count_frames(recording)),
                    "frame_rate": str(self.frame_rate),
                    "details": self._details(episode),
                },
                headers={"X-API-Key": api_key},
                timeout=TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        logger.info(f"Beaten {level} uploaded from {recording.name}.")
