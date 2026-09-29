from typing import Any

from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.story_book import StoryBook, label

QUIET_STEPS = 100

Event = dict[str, str]


def event(kind: str, text: str, detail: str) -> Event:
    return {"kind": kind, "text": text, "detail": detail}


class StoryTeller:
    def __init__(self, book: StoryBook) -> None:
        self.book = book
        self.step = 0
        self.last_change: dict[str, int] = {}
        self.facts: dict[str, Any] | None = None
        self.location: tuple[int, int] | None = None

    def observe(self, transition: Transition, attempt: int) -> list[Event]:
        self.step += 1
        info = transition.info
        if self.facts is None and not self.book.has_reached(info["state"]):
            self.book.reach(info["state"], attempt)
        events = self._fact_events(info["ram"])
        if info["location"] is not None:
            self.location = tuple(info["location"])
        if info["state_transition"] is not None:
            events.append(self._transition_event(*info["state_transition"], attempt))
        return events

    def finish(self, episode: EpisodeRecord, new_best: bool, last_image: str) -> list[Event]:
        succeeded = episode.curriculum_succeeded or episode.won
        failures_today = self.book.finish(episode.curriculum_state, succeeded, self.location, last_image)
        self.book.save()
        self.facts, self.location = None, None
        events = []
        if episode.won:
            events.append(event("milestone", "Level cleared!", f"{episode.score:.1f} points"))
        elif not succeeded:
            events.append(event("bad", f"Attempt over in {label(episode.final_state)}", f"#{failures_today} today"))
        if new_best:
            events.append(event("good", "New best score!", f"{episode.score:.1f} points"))
        return events

    def _transition_event(self, previous: str, current: str, attempt: int) -> Event:
        if self.book.phases.index(current) < self.book.phases.index(previous):
            return event("bad", f"Back to {label(current)}", f"Lost progress in {label(previous)}")
        if self.book.has_reached(current):
            return event("good", f"{label(previous)} done", f"Next: {label(current)}")
        self.book.reach(current, attempt)
        return event("milestone", f"New area: {label(current)}", f"First time, attempt #{attempt}")

    def _fact_events(self, facts: dict[str, Any]) -> list[Event]:
        previous, self.facts = self.facts, facts
        if previous is None:
            return []
        events = []
        for name, value in facts.items():
            if name not in previous or value == previous[name] or not isinstance(value, int | bool):
                continue
            quiet = name not in self.last_change or self.step - self.last_change[name] >= QUIET_STEPS
            self.last_change[name] = self.step
            if quiet:
                events.append(self._fact_event(name, previous[name], value))
        return events

    @staticmethod
    def _fact_event(name: str, before: int | bool, after: int | bool) -> Event:
        if isinstance(after, bool):
            return event("good" if after else "bad", f"{label(name)}: {'yes' if after else 'no'}", "")
        rising = after > before
        return event("good" if rising else "bad", f"{label(name)} {'up' if rising else 'down'} to {after}", "")
