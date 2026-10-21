import json
from pathlib import Path
from typing import Any

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
        self.published: set[str] | None = None
        self.uploaded: set[str] = set()

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        pass

    def on_update(self) -> None:
        if self.settings.api_key is None:
            return
        if not self.context.config.paths.reports_dir.is_dir():
            return
        try:
            if self.published is None:
                self.published = self._published()
        except httpx.HTTPError as error:
            logger.error(f"Published lab reports unavailable: {error}")
            return
        for source in self._sources(self.published):
            self._upload(source)

    def _published(self) -> set[str]:
        response = httpx.get(f"{self.settings.url}/reports", timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return {report["name"] for report in response.json() if report["game"] == self.context.game}

    def _sources(self, published: set[str]) -> list[Path]:
        if self.digest is not None:
            return self.digest.due(published)
        return sorted(self.context.config.paths.reports_dir.glob(f"*{SUMMARY_SUFFIX}"))

    def _summary(self, source: Path) -> dict[str, Any]:
        if self.digest is not None:
            return self.digest.of(source)
        return json.loads(source.read_text(encoding="utf-8"))

    def _upload(self, source: Path) -> None:
        try:
            self._upload_once(self._summary(source))
        except httpx.HTTPError as error:
            logger.error(f"Lab report {source.name} upload failed: {error}")
        except (KeyError, IndexError, ValueError) as error:
            logger.warning(f"No summary of lab report {source.name}: {error}")

    def _upload_once(self, summary: dict[str, Any]) -> None:
        version = json.dumps(summary, sort_keys=True)
        if version in self.uploaded:
            return
        name = summary["name"]
        response = httpx.put(
            f"{self.settings.url}/reports/{self.context.game}/{name}",
            json={"headline": summary["headline"], "lines": summary["lines"]},
            headers={"X-API-Key": self.settings.api_key},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        self.uploaded.add(version)
        self.published = (self.published or set()) | {name}
        logger.info(f"Short lab report {name} uploaded.")
