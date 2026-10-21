from datenwissenschaften.tracking.run import TrackingRun
from datenwissenschaften.training.checkpoint_hook import learning_metadata
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels


class TrackingHook:
    def __init__(self, run: TrackingRun, models: StateModels, phases: tuple[str, ...]) -> None:
        self.run = run
        self.models = models
        self.phases = phases
        self.updates = 0

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        metrics = {
            "progress": float(self.phases.index(episode.final_state)),
            "success": float(episode.curriculum_succeeded),
            "score": episode.score,
        }
        if episode.duration_seconds > 0:
            metrics["decisions_per_second"] = episode.step_count / episode.duration_seconds
        self.run.log(metrics, episode.episode_index)

    def on_update(self) -> None:
        self.updates += 1
        self.run.log({"confidence": 1.0 - learning_metadata(self.models)["entropy_share"]}, self.updates)
