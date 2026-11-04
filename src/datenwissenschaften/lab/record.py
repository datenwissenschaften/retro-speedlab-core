from pathlib import Path

import httpx
import typer
from loguru import logger

from datenwissenschaften.lab.timeline import timeline
from datenwissenschaften.lab.tracing import LabRun, run_scores, send_trace
from datenwissenschaften.lab.transcript import Round, read_transcript
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.ui.relay import relay_key

TIMEOUT_SECONDS = 60.0
COUNTED = ("turns", "subagents", "commits", "tool_errors")


def summary(run: LabRun, rounds: list[Round]) -> dict[str, object]:
    scores = run_scores(run, rounds)
    counted = {name: int(scores[name]) for name in COUNTED}
    measured = {"cost_usd": round(scores["cost_usd"], 4), "minutes": round(scores["minutes"], 2)}
    return {"outcome": run.outcome, **measured, **counted, "timeline": timeline(rounds)}


def publish_lab_run(settings: RetroSpeedlabConfig, run: LabRun, rounds: list[Round]) -> None:
    if not settings.ui.enabled:
        return
    headers = {"X-API-Key": relay_key(settings.upload)}
    with httpx.Client(base_url=settings.upload.url, headers=headers, timeout=TIMEOUT_SECONDS) as client:
        try:
            client.put(f"/lab-runs/{run.game}/{run.stamp}", json=summary(run, rounds)).raise_for_status()
        except httpx.HTTPError as error:
            logger.warning(f"The lab run {run.stamp} was not published: {error}")
            return
    logger.info(f"The lab run {run.stamp} is published")


app = typer.Typer()


@app.command()
def record_lab_run(config: Path, log: Path, stamp: str, base: str, outcome: str, commits: int) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    run = LabRun(settings.training.game, stamp, base, outcome, commits)
    rounds = read_transcript(log)
    if not rounds:
        raise ValueError(f"the lab log {log} has no finished round")
    publish_lab_run(settings, run, rounds)
    send_trace(run, rounds)


if __name__ == "__main__":
    app()
