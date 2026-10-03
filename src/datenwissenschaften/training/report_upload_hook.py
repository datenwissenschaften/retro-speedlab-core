import json

import httpx
from loguru import logger

from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.report_digest import SUMMARY_SUFFIX

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
        reports_dir = self.context.config.paths.reports_dir
        if not reports_dir.is_dir():
            return
        try:
            for summary in sorted(reports_dir.glob(f"*{SUMMARY_SUFFIX}")):
                version = (summary.name, summary.stat().st_mtime)
                if version in self.uploaded:
                    continue
                self._upload(json.loads(summary.read_text(encoding="utf-8")), self.settings.api_key)
                self.uploaded.add(version)
        except httpx.HTTPError as error:
            logger.error(f"Lab report upload failed: {error}")

    def _upload(self, summary: dict[str, str | list[str]], api_key: str) -> None:
        response = httpx.put(
            f"{self.settings.url}/reports/{self.context.game}/{summary['name']}",
            json={"headline": summary["headline"], "lines": summary["lines"]},
            headers={"X-API-Key": api_key},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        logger.info(f"Short lab report {summary['name']} uploaded.")
