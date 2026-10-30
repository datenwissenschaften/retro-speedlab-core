from pathlib import Path

import httpx
from loguru import logger

from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.ui.relay import relay_key

REPORTS_DIR = Path("agent") / "reports"
REPORT_SUFFIX = ".md"
TIMEOUT_SECONDS = 120.0


def unpublished(client: httpx.Client, game: str, project: Path) -> list[Path]:
    response = client.get("/reports")
    response.raise_for_status()
    published = {report["name"] for report in response.json() if report["game"] == game}
    return [path for path in sorted((project / REPORTS_DIR).glob(f"*{REPORT_SUFFIX}")) if path.name not in published]


def publish(client: httpx.Client, game: str, title: str, project: Path) -> None:
    for report in unpublished(client, game, project):
        body = {"title": title, "text": report.read_text(encoding="utf-8")}
        try:
            client.put(f"/reports/{game}/{report.name}", json=body).raise_for_status()
        except httpx.HTTPStatusError as error:
            logger.warning(f"Lab report {report.name} was not published: {error}")
            continue
        logger.info(f"Lab report {report.name} published")


def publish_reports(settings: RetroSpeedlabConfig, project: Path) -> None:
    if not settings.ui.enabled:
        return
    headers = {"X-API-Key": relay_key(settings.upload)}
    with httpx.Client(base_url=settings.upload.url, headers=headers, timeout=TIMEOUT_SECONDS) as client:
        try:
            publish(client, settings.training.game, settings.speedrun.game.title, project)
        except httpx.HTTPError as error:
            logger.warning(f"The lab reports wait for the website: {error}")
