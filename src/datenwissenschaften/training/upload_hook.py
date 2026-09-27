import json
from pathlib import Path

import httpx
from itsdangerous import Signer
from loguru import logger

from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.serialization import to_json_value
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.system import system_metadata

TIMEOUT_SECONDS = 30


class UploadHook:
    def __init__(self, context: RunContext, agent: LayaAgent) -> None:
        self.context = context
        self.agent = agent
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
            signed_metadata = self._signed_metadata(self.settings.api_key)
            for episode in list(self.pending):
                self._upload(episode, signed_metadata, self.settings.api_key)
                self.pending.remove(episode)
        except httpx.HTTPError as error:
            logger.error(f"Episode upload failed: {error}")

    def _signed_metadata(self, api_key: str) -> bytes:
        response = httpx.get(
            f"{self.settings.url}/runs/signing-key", headers={"X-API-Key": api_key}, timeout=TIMEOUT_SECONDS
        )
        response.raise_for_status()
        metadata = {
            **self.agent.metadata(),
            "game": self.context.game,
            "savestate": self.context.savestate,
            "system": system_metadata(),
        }
        metadata_json = json.dumps(to_json_value(metadata), indent=4, sort_keys=True)
        return Signer(response.json()["signing_key"]).sign(metadata_json.encode("utf-8"))

    def _upload(self, episode: EpisodeRecord, signed_metadata: bytes, api_key: str) -> None:
        recording = Path(episode.bk2_path)
        with recording.open("rb") as recording_file:
            response = httpx.post(
                f"{self.settings.url}/runs",
                files={
                    "bk2_file": (recording.name, recording_file, "application/octet-stream"),
                    "metadata_file": ("metadata.json.signed", signed_metadata, "application/octet-stream"),
                },
                data={"game": self.context.game, "category": self.context.savestate},
                headers={"X-API-Key": api_key},
                timeout=TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        logger.info(f"Episode {recording.name} uploaded successfully.")
