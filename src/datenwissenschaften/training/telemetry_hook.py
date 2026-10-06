from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.telemetry import publish_episode, publish_metadata

MEMORY_PUBLISH_STEPS = 10


class TelemetryHook:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.steps = 0

    def on_step(self, transition: Transition) -> None:
        self.steps += 1
        if self.steps % MEMORY_PUBLISH_STEPS != 1:
            return
        info = transition.info
        publish_metadata("memory", {"state": info["state"], "fields": info["memory"]}, replace=True)

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
