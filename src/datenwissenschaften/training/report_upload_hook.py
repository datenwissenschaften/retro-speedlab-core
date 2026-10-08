import json
from pathlib import Path

import httpx
from loguru import logger

from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.summaries import SUMMARY_SUFFIX, ReportDigest

TIMEOUT_SECONDS = 30


class ReportUploadHook:
    def __init__(self, context: RunContext, digest: ReportDigest | None) -> None:
        self.context = context
        self.digest = digest
        self.settings = context.config.upload
        self.uploaded: set[tuple[str, float]] = set()

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        if self.settings.api_key is None:
            return
        reports_dir = self.context.config.paths.reports_dir
        if not reports_dir.is_dir():
            return
        self._summarize_latest()
        try:
            for summary in sorted(reports_dir.glob(f"*{SUMMARY_SUFFIX}")):
                self._upload_once(summary, json.loads(summary.read_text(encoding="utf-8"))["name"])
        except httpx.HTTPError as error:
            logger.error(f"Lab report upload failed: {error}")

    def _summarize_latest(self) -> None:
        if self.digest is None:
            return
        try:
            self.digest.latest()
        except FileNotFoundError:
            return
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as error:
            logger.warning(f"No summary of the latest lab report: {error}")

    def _upload_once(self, summary: Path, name: str) -> None:
        version = (str(summary), summary.stat().st_mtime)
        if version in self.uploaded:
            return
        content = json.loads(summary.read_text(encoding="utf-8"))
        response = httpx.put(
            f"{self.settings.url}/reports/{self.context.game}/{name}",
            json={"headline": content["headline"], "lines": content["lines"]},
            headers={"X-API-Key": self.settings.api_key},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        self.uploaded.add(version)
        logger.info(f"Short lab report {name} uploaded.")
