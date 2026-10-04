import json
from pathlib import Path

import httpx
from loguru import logger

from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.summaries import SUMMARY_SUFFIX, summary_path

TIMEOUT_SECONDS = 30


class ReportUploadHook:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.settings = context.config.upload
        self.uploaded: set[tuple[str, float]] = set()

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        if self.settings.api_key is None:
            return
        paths = self.context.config.paths
        reports = sorted(paths.reports_dir.glob(f"*{SUMMARY_SUFFIX}")) if paths.reports_dir.is_dir() else []
        hints = summary_path(paths.hints_file)
        try:
            for summary in reports:
                self._upload_once(summary, f"reports/{self.context.game}/{json.loads(summary.read_text())['name']}")
            if hints.is_file():
                self._upload_once(hints, f"hints/{self.context.game}")
        except httpx.HTTPError as error:
            logger.error(f"Lab summary upload failed: {error}")

    def _upload_once(self, summary: Path, route: str) -> None:
        version = (str(summary), summary.stat().st_mtime)
        if version in self.uploaded:
            return
        content = json.loads(summary.read_text(encoding="utf-8"))
        response = httpx.put(
            f"{self.settings.url}/{route}",
            json={"headline": content["headline"], "lines": content["lines"]},
            headers={"X-API-Key": self.settings.api_key},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        self.uploaded.add(version)
        logger.info(f"Lab summary {summary.name} uploaded.")
