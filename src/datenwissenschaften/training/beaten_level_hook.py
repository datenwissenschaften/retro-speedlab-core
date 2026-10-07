from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.ui.live import live_feed


class BeatenLevelHook:
    def __init__(self, curriculum: CurriculumRun) -> None:
        self.curriculum = curriculum

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        for level, members in self.curriculum.targets.levels.items():
            if self.curriculum.curriculum.is_mastered(level):
                live_feed.drop_replays(members)

    def on_update(self) -> None:
        pass
