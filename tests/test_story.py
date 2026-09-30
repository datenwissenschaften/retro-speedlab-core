from pathlib import Path

import numpy as np
import pytest

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.training import story_book
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.story_book import DANGER_CELL, StoryBook, label, level_identity, story_key
from datenwissenschaften.training.story_teller import QUIET_STEPS, StoryTeller

PHASES = ("FindDoor", "OpenDoor")


@pytest.fixture
def published(monkeypatch) -> list[dict]:
    views = []
    monkeypatch.setattr(story_book, "publish_metadata", lambda section, values: views.append((section, values)))
    return views


def _transition(ram: dict, location: tuple[int, int], state: str, change: tuple[str, str] | None) -> Transition:
    info = {"state": state, "ram": ram, "location": location, "state_transition": change}
    observation = {"state": "{}", "question": "Which move?"}
    return Transition(1, observation, Decision(0, {"left": 1.0}, 1.0), [np.zeros((2, 2, 3))], 0.0, False, info)


def _episode(phase: str, succeeded: bool) -> EpisodeRecord:
    info = {
        "episode_bk2_path": "run.bk2",
        "started_from_initial_savestate": True,
        "episode_start_state": "Level1",
        "episode_start_score": 0.0,
        "state": phase,
    }
    episode = EpisodeRecord.start(0, info)
    episode.curriculum_succeeded = succeeded
    return episode


def test_labels_read_like_words():
    assert (label("FindOpenDoor"), label("nibbleys_eaten")) == ("Find Open Door", "Nibbleys Eaten")


def test_first_visit_of_a_phase_is_a_milestone_and_later_ones_are_progress(tmp_path: Path, published):
    teller = StoryTeller(StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES))

    teller.observe(_transition({"weight": 1}, (0, 0), "FindDoor", None), 1)
    first = teller.observe(_transition({"weight": 1}, (0, 0), "OpenDoor", ("FindDoor", "OpenDoor")), 1)
    again = teller.observe(_transition({"weight": 1}, (0, 0), "OpenDoor", ("FindDoor", "OpenDoor")), 2)

    assert first == [{"kind": "milestone", "text": "New area: Open Door", "detail": "First time, attempt #1"}]
    assert again[0]["kind"] == "good"


def test_falling_back_to_an_earlier_phase_is_told_as_a_setback(tmp_path: Path, published):
    teller = StoryTeller(StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES))
    teller.observe(_transition({"weight": 1}, (0, 0), "OpenDoor", None), 1)

    back = teller.observe(_transition({"weight": 1}, (0, 0), "FindDoor", ("OpenDoor", "FindDoor")), 1)

    assert back == [{"kind": "bad", "text": "Back to Find Door", "detail": "Lost progress in Open Door"}]


def test_only_quiet_facts_become_events(tmp_path: Path, published):
    teller = StoryTeller(StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES))
    teller.observe(_transition({"weight": 1, "time": 90, "door": False}, (0, 0), "FindDoor", None), 1)

    changed = teller.observe(_transition({"weight": 2, "time": 89, "door": True}, (0, 0), "FindDoor", None), 1)
    ticking = teller.observe(_transition({"weight": 2, "time": 88, "door": True}, (0, 0), "FindDoor", None), 1)

    assert [event["text"] for event in changed] == ["Weight up to 2", "Time down to 89", "Door: yes"]
    assert ticking == []
    for _ in range(QUIET_STEPS):
        teller.observe(_transition({"weight": 2, "time": 88, "door": True}, (0, 0), "FindDoor", None), 1)
    assert teller.observe(_transition({"weight": 2, "time": 87, "door": True}, (0, 0), "FindDoor", None), 1)


def test_failures_are_counted_and_persisted(tmp_path: Path, published):
    database = JsonDatabase(tmp_path / "db.json")
    teller = StoryTeller(StoryBook(database, "Game", "Level1", PHASES))
    teller.observe(_transition({}, (40, 20), "FindDoor", None), 1)

    first = teller.finish(_episode("FindDoor", False), False, "jpeg-1")
    teller.observe(_transition({}, (40, 20), "FindDoor", None), 2)
    second = teller.finish(_episode("FindDoor", True), True, "jpeg-2")

    view = StoryBook(database, "Game", "Level1", PHASES).view()
    assert first == [{"kind": "bad", "text": "Attempt over in Find Door", "detail": "#1 today"}]
    assert second == [{"kind": "good", "text": "New best score!", "detail": "0.0 points"}]
    assert view["phases"][0] == {
        "name": "FindDoor",
        "label": "Find Door",
        "reached": True,
        "first_attempt": 1,
    }
    assert view["phases"][1]["reached"] is False
    assert view["danger"] == [{"phase": "Find Door", "count": 1, "image": "jpeg-1"}]
    assert published[-1] == ("stories", {"Level1": view})
    assert database.contains(story_key(level_identity("Game", "Level1")))


def test_danger_spots_rank_places_by_recent_failures_with_their_latest_picture(tmp_path: Path, published):
    book = StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES)
    near, far = (10, 10), (10 + 4 * DANGER_CELL, 10)

    for index in range(3):
        book.finish("FindDoor", False, near, f"near-{index}")
    book.finish("FindDoor", False, far, "far")
    book.finish("OpenDoor", False, None, "door")

    assert [(spot["count"], spot["image"]) for spot in book.view()["danger"]] == [
        (3, "near-2"),
        (1, "far"),
        (1, "door"),
    ]


def test_a_story_in_an_older_format_is_replaced(tmp_path: Path, published):
    database = JsonDatabase(tmp_path / "db.json")
    database.set(story_key(level_identity("Game", "Level1")), {"reached": {"FindDoor": 3}, "visits": {}, "ends": {}})

    view = StoryBook(database, "Game", "Level1", PHASES).view()

    assert view["danger"] == []
    assert view["phases"][0]["reached"] is False
