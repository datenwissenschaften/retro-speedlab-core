from pathlib import Path

import numpy as np
import pytest

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.training import story_book
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.story_book import DANGER_CELL, StoryBook, label, level_identity, story_key
from datenwissenschaften.training.story_teller import StoryTeller

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


def test_every_phase_remembers_the_attempt_that_first_reached_it(tmp_path: Path, published):
    book = StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES)
    teller = StoryTeller(book)

    teller.observe(_transition({}, (0, 0), "FindDoor", None), 1)
    teller.observe(_transition({}, (0, 0), "OpenDoor", ("FindDoor", "OpenDoor")), 2)
    teller.observe(_transition({}, (0, 0), "OpenDoor", ("FindDoor", "OpenDoor")), 3)

    assert [phase["first_attempt"] for phase in book.view()["phases"]] == [1, 2]


def test_failures_are_counted_and_persisted(tmp_path: Path, published):
    database = JsonDatabase(tmp_path / "db.json")
    teller = StoryTeller(StoryBook(database, "Game", "Level1", PHASES))
    teller.observe(_transition({}, (40, 20), "FindDoor", None), 1)

    teller.finish(_episode("FindDoor", False), "jpeg-1")
    teller.observe(_transition({}, (40, 20), "FindDoor", None), 2)
    teller.finish(_episode("FindDoor", True), "jpeg-2")

    view = StoryBook(database, "Game", "Level1", PHASES).view()
    assert view["phases"][0] == {
        "name": "FindDoor",
        "label": "Find Door",
        "reached": True,
        "first_attempt": 1,
    }
    assert view["phases"][1]["reached"] is False
    assert view["failures"] == 1
    assert view["danger"] == [{"phase": "Find Door", "located": True, "count": 1, "image": "jpeg-1"}]
    assert published[-1] == ("stories", {"Level1": view})
    assert database.contains(story_key(level_identity("Game", "Level1")))


def test_danger_spots_rank_places_by_recent_failures_with_their_latest_picture(tmp_path: Path, published):
    book = StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES)
    near, far = (10, 10), (10 + 4 * DANGER_CELL, 10)

    for index in range(3):
        book.finish("FindDoor", False, near, f"near-{index}")
    book.finish("FindDoor", False, far, "far")
    book.finish("OpenDoor", False, None, "door")

    view = book.view()
    assert view["failures"] == 5
    assert [(spot["count"], spot["located"], spot["image"]) for spot in view["danger"]] == [
        (3, True, "near-2"),
        (1, True, "far"),
        (1, False, "door"),
    ]


def test_the_danger_window_keeps_only_the_most_recent_failures(tmp_path: Path, published):
    book = StoryBook(JsonDatabase(tmp_path / "db.json"), "Game", "Level1", PHASES)

    for index in range(story_book.DANGER_WINDOW + 5):
        book.finish("FindDoor", False, None, f"frame-{index}")

    view = book.view()
    assert view["failures"] == story_book.DANGER_WINDOW
    assert view["danger"] == [
        {
            "phase": "Find Door",
            "located": False,
            "count": story_book.DANGER_WINDOW,
            "image": f"frame-{story_book.DANGER_WINDOW + 4}",
        }
    ]


def test_a_story_in_an_older_format_is_replaced(tmp_path: Path, published):
    database = JsonDatabase(tmp_path / "db.json")
    database.set(story_key(level_identity("Game", "Level1")), {"reached": {"FindDoor": 3}, "visits": {}, "ends": {}})

    view = StoryBook(database, "Game", "Level1", PHASES).view()

    assert view["danger"] == []
    assert view["phases"][0]["reached"] is False
