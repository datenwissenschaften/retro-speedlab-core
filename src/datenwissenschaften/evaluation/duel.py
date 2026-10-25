import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import msgspec
import typer
from loguru import logger
from scipy.stats import mannwhitneyu

from datenwissenschaften.evaluation import branch_play
from datenwissenschaften.evaluation.branch_play import BranchPlay
from datenwissenschaften.evaluation.player import Attempt
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tracking.run import TrackingRun, tracking_client

AGENTS = ("claude", "codex")
DRAW = "draw"
EXPERIMENT_SUFFIX = " A/B"
COIN_FLIP = 0.5
CERTAIN = 1.0


@dataclass(slots=True, frozen=True)
class Entrant:
    agent: str
    start: str
    attempts: list[Attempt]

    @property
    def wins(self) -> int:
        return sum(attempt.won for attempt in self.attempts)

    @property
    def progress(self) -> list[float]:
        return [attempt.progress for attempt in self.attempts]

    def rank(self) -> tuple[int, float, float]:
        winning = [attempt.decisions for attempt in self.attempts if attempt.won]
        return self.wins, mean(self.progress), -mean(winning) if winning else 0.0

    def metrics(self) -> dict[str, float]:
        return {
            f"{self.agent}/wins": float(self.wins),
            f"{self.agent}/progress": mean(self.progress),
            f"{self.agent}/decisions": mean(attempt.decisions for attempt in self.attempts),
        }


@dataclass(slots=True, frozen=True)
class Verdict:
    winner: str
    confidence: float


def probability_better(worse: list[float], better: list[float]) -> float:
    return float(mannwhitneyu(better, worse, alternative="greater").statistic / (len(better) * len(worse)))


def decide(entrants: list[Entrant]) -> Verdict:
    if not entrants:
        raise RuntimeError("Every branch crashed, so the duel has no winner.")
    if len(entrants) == 1:
        return Verdict(entrants[0].agent, CERTAIN)
    best, other = sorted(entrants, key=Entrant.rank, reverse=True)
    if best.rank() == other.rank():
        return Verdict(DRAW, COIN_FLIP)
    return Verdict(best.agent, probability_better(other.progress, best.progress))


def branch_entrant(wrapper: str, config: Path, project: Path, agent: str, output_dir: Path) -> Entrant | None:
    branch = project.with_name(f"{project.name}-{agent}")
    output = output_dir / f"{agent}.json"
    command = [sys.executable, "-m", branch_play.__name__, wrapper, str(config), str(branch), str(output)]
    try:
        subprocess.run(command, cwd=branch, stdout=sys.stderr, check=True)
    except subprocess.CalledProcessError as error:
        logger.error(f"The {agent} branch crashed while playing and forfeits the duel: {error}")
        return None
    played = msgspec.json.decode(output.read_bytes(), type=BranchPlay)
    return Entrant(agent, played.start, played.attempts)


def record(settings: RetroSpeedlabConfig, entrants: list[Entrant], verdict: Verdict) -> None:
    client = tracking_client(settings.tracking)
    if client is None:
        raise RuntimeError("A duel is always tracked, but tracking.token is not set.")
    experiment = f"{settings.training.game_identity}{EXPERIMENT_SUFFIX}"
    starts = {f"{entrant.agent}/start": entrant.start for entrant in entrants}
    parameters = {"winner": verdict.winner, "episodes": str(branch_play.EPISODES), **starts}
    run = TrackingRun.start(client, experiment, settings.ui.release, parameters)
    metrics = {key: value for entrant in entrants for key, value in entrant.metrics().items()}
    run.log({**metrics, "confidence": verdict.confidence}, 0)
    run.finish()


def judge(wrapper: str, config: Path, project: Path) -> None:
    settings = load_config(config)
    output_dir = settings.paths.cache_dir / "duel"
    played = [branch_entrant(wrapper, config, project, agent, output_dir) for agent in AGENTS]
    entrants = [entrant for entrant in played if entrant is not None]
    verdict = decide(entrants)
    for entrant in entrants:
        logger.info(
            f"Duel {entrant.agent} from {entrant.start}: {entrant.wins} wins, progress {mean(entrant.progress):.2f}"
        )
    logger.info(f"Duel winner: {verdict.winner} with confidence {verdict.confidence:.2f}")
    record(settings, entrants, verdict)
    typer.echo(verdict.winner)


if __name__ == "__main__":
    typer.run(judge)
