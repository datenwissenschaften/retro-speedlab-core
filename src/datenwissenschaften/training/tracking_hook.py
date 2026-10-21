from collections import Counter

from datenwissenschaften.tracking.run import TrackingRun
from datenwissenschaften.training.checkpoint_hook import learning_metadata
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels

WHOLE_LEVEL = "whole level"


def chart_group(part: str, levels: dict[str, tuple[str, ...]]) -> str:
    if part in levels:
        return f"{part}/{WHOLE_LEVEL}"
    for level, states in levels.items():
        if part in states:
            return f"{level}/{part}"
    return part


class TrackingHook:
    def __init__(
        self, run: TrackingRun, models: StateModels, phases: tuple[str, ...], levels: dict[str, tuple[str, ...]]
    ) -> None:
        self.run = run
        self.models = models
        self.phases = phases
        self.levels = levels
        self.attempts: Counter[str] = Counter()
        self.updates = 0

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        group = chart_group(episode.curriculum_state, self.levels)
        self.attempts[group] += 1
        metrics = {
            f"{group}/progress": float(self.phases.index(episode.final_state)),
            f"{group}/success": float(episode.curriculum_succeeded),
            f"{group}/score": episode.score,
        }
        self.run.log(metrics, self.attempts[group])
        if episode.duration_seconds > 0:
            self.run.log({"decisions_per_second": episode.step_count / episode.duration_seconds}, episode.episode_index)

    def on_update(self) -> None:
        self.updates += 1
        self.run.log({"confidence": 1.0 - learning_metadata(self.models)["entropy_share"]}, self.updates)
