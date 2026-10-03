import json
import subprocess
from pathlib import Path

import httpx
from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.serialization import to_json_value
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.system import system_metadata
from datenwissenschaften.training.video_render import count_frames, render_video

TIMEOUT_SECONDS = 120


class UploadHook:
    def __init__(self, context: RunContext, agent: LayaAgent, frame_rate: float) -> None:
        self.context = context
        self.agent = agent
        self.frame_rate = frame_rate
        self.settings = context.config.upload
        self.pending: list[EpisodeRecord] = []

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        if not episode.won:
            return
        if not episode.started_from_initial_savestate:
            logger.info("Not uploading a curriculum-checkpoint win; only complete runs are accepted.")
            return
        self.pending.append(episode)

    def on_update(self) -> None:
        if not self.pending:
            return
        if self.settings.api_key is None:
            logger.info("Upload API key is not configured. Skipping episode upload.")
            self.pending.clear()
            return
        try:
            for episode in list(self.pending):
                self._upload(episode, self.settings.api_key)
                self.pending.remove(episode)
        except (httpx.HTTPError, subprocess.CalledProcessError) as error:
            logger.error(f"Episode upload failed: {error}")

    def _details(self, episode: EpisodeRecord) -> str:
        details = {
            "laya": self.agent.metadata(),
            "system": system_metadata(),
            "episode": episode.episode_index,
            "steps": episode.step_count,
            "score": episode.score,
        }
        return json.dumps(to_json_value(details), sort_keys=True)

    def _upload(self, episode: EpisodeRecord, api_key: str) -> None:
        recording = Path(episode.bk2_path)
        video = render_video(self.context.config.paths.roms_path, recording)
        with video.open("rb") as video_file:
            response = httpx.post(
                f"{self.settings.url}/runs",
                files={"video": (video.name, video_file, "video/mp4")},
                data={
                    "game": self.context.game,
                    "level": self.context.savestate,
                    "frames": str(count_frames(recording)),
                    "frame_rate": str(self.frame_rate),
                    "details": self._details(episode),
                },
                headers={"X-API-Key": api_key},
                timeout=TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        logger.info(f"Beaten level {self.context.savestate} uploaded from {recording.name}.")
