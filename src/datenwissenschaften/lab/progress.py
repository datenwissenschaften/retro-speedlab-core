from pathlib import Path
from typing import Any

import httpx
import typer
from loguru import logger
from opentelemetry.sdk.trace import TracerProvider

from datenwissenschaften.lab.tracing import SESSION_PREFIX, langfuse_client, missing_langfuse_keys
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config

TIMEOUT_SECONDS = 60.0
PROGRESSED = "progressed"


def progress_scores(progress: dict[str, Any]) -> dict[str, float]:
    scores = {PROGRESSED: float(progress[PROGRESSED]), "run_seconds_change": float(progress["seconds"])}
    if progress["hours"] is not None:
        scores["hours_to_progress"] = float(progress["hours"])
    return scores


def measured_runs(settings: RetroSpeedlabConfig) -> list[dict[str, Any]]:
    with httpx.Client(base_url=settings.upload.url, timeout=TIMEOUT_SECONDS) as client:
        response = client.get("/lab-runs")
        response.raise_for_status()
    game = settings.speedrun.game.id
    return [run for run in response.json() if run["game"] == game and run["progress"] is not None]


def score_progress(runs: list[dict[str, Any]]) -> None:
    missing = missing_langfuse_keys()
    if missing:
        logger.warning(f"The progress of {len(runs)} lab runs is not scored: {', '.join(missing)} not set")
        return
    langfuse = langfuse_client(TracerProvider())
    for run in runs:
        session = f"{SESSION_PREFIX}{run['stamp']}"
        for name, value in progress_scores(run["progress"]).items():
            data_type = "BOOLEAN" if name == PROGRESSED else "NUMERIC"
            score_id = f"progress-{run['stamp']}-{name}"
            langfuse.create_score(name=name, value=value, session_id=session, score_id=score_id, data_type=data_type)
    langfuse.shutdown()
    logger.info(f"The progress of {len(runs)} lab runs is scored in Langfuse")


def score_previous_run(settings: RetroSpeedlabConfig, stamp: str) -> None:
    if not settings.ui.enabled:
        return
    try:
        earlier = [run for run in measured_runs(settings) if run["stamp"] < stamp]
    except httpx.HTTPError as error:
        logger.warning(f"The progress before lab run {stamp} was not read: {error}")
        return
    score_progress(earlier[:1])


app = typer.Typer()


@app.command()
def score_all_progress(config: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    score_progress(measured_runs(settings))


if __name__ == "__main__":
    app()
