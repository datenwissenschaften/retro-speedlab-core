from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.telemetry import publish_episode


class TelemetryHook:
    def __init__(self, context: RunContext) -> None:
        self.context = context

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        publish_episode(
            savestate=self.context.savestate,
            training_state=episode.curriculum_state,
            fitness=episode.score,
            training_steps=episode.step_count,
            total_steps=episode.step_count,
            duration_seconds=episode.duration_seconds,
            won=episode.won,
            started_from_initial_savestate=episode.started_from_initial_savestate,
            final_state=episode.final_state,
        )

    def on_update(self) -> None:
        pass
