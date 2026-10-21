from datenwissenschaften.tracking.run import TrackingRun
from datenwissenschaften.training.checkpoint_hook import learning_metadata
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels


class TrackingHook:
    def __init__(self, run: TrackingRun, models: StateModels) -> None:
        self.run = run
        self.models = models

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        metrics = {
            "episode/score": episode.score,
            "episode/steps": float(episode.step_count),
            "episode/seconds": episode.duration_seconds,
            "episode/won": float(episode.won),
            "episode/curriculum_succeeded": float(episode.curriculum_succeeded),
            f"score/{episode.curriculum_state}": episode.score,
        }
        self.run.log(metrics, episode.episode_index)

    def on_update(self) -> None:
        state = self.models.require_active()
        learning = learning_metadata(self.models)
        steps = int(learning.pop("num_timesteps"))
        self.run.log({f"{state}/{name}": float(value) for name, value in learning.items()}, steps)
