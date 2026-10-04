from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.story_book import StoryBook


class StoryTeller:
    def __init__(self, book: StoryBook) -> None:
        self.book = book
        self.location: tuple[int, int] | None = None

    def observe(self, transition: Transition, attempt: int) -> None:
        info = transition.info
        if not self.book.has_reached(info["state"]):
            self.book.reach(info["state"], attempt)
        if info["location"] is not None:
            self.location = tuple(info["location"])

    def finish(self, episode: EpisodeRecord, last_image: str) -> None:
        succeeded = episode.curriculum_succeeded or episode.won
        self.book.finish(episode.curriculum_state, succeeded, self.location, last_image)
        self.book.save()
        self.location = None
