import pytest

from datenwissenschaften.callbacks.episode_record import EpisodeRecord, _require_won


def test_require_won_raises_when_missing():
    with pytest.raises(ValueError, match="must include"):
        _require_won({})


def test_require_won_raises_when_not_bool():
    with pytest.raises(ValueError, match="must be a boolean"):
        _require_won({"won": "yes"})


def test_add_step_captures_metadata_from_first_step_only():
    episode = EpisodeRecord(0, 0)

    episode.add_step(
        {
            "won": False,
            "episode_bk2_path": "run-000001.bk2",
            "started_from_initial_savestate": True,
            "curriculum_state": "FindDispenser",
            "episode_start_state": "Idle",
        }
    )
    episode.add_step(
        {
            "won": False,
            "episode_bk2_path": "run-000002.bk2",
            "started_from_initial_savestate": False,
            "curriculum_state": "EatFood",
            "episode_start_state": "Eating",
        }
    )

    assert episode.bk2_path == "run-000001.bk2"
    assert episode.started_from_initial_savestate is True
    assert episode.curriculum_state == "FindDispenser"
    assert episode.episode_start_state == "Idle"
    assert episode.step_count == 2


def test_add_step_marks_won_once_and_keeps_it_sticky():
    episode = EpisodeRecord(0, 0)

    episode.add_step({"won": True})
    episode.add_step({"won": False})

    assert episode.won is True
