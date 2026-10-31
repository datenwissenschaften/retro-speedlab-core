from pathlib import Path

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.shown import SHOWN_FILE, worth_showing

STATES = ("Play", "Fed", "Heavy", "Won")


def piece(state: str, length: int) -> Segment:
    return Segment(state, [[0]] * length, b"", False)


def test_the_stream_keeps_the_faster_movie_when_the_search_takes_a_slower_way_into_the_same_state(tmp_path: Path):
    shown = tmp_path / SHOWN_FILE

    assert worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 200)])
    assert not worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 300)])
    assert worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 200)])


def test_a_movie_that_gets_further_is_shown_even_when_it_is_slower(tmp_path: Path):
    shown = tmp_path / SHOWN_FILE
    worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 200)])

    assert worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 400), piece("Won", 90)])


def test_a_checkpoint_in_the_same_state_is_no_new_piece(tmp_path: Path):
    shown = tmp_path / SHOWN_FILE
    worth_showing(shown, STATES, "Play", [piece("Fed", 100)])

    assert not worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Fed", 20)])


def test_a_game_package_with_other_states_shows_its_first_movie(tmp_path: Path):
    shown = tmp_path / SHOWN_FILE
    worth_showing(shown, STATES, "Play", [piece("Fed", 100), piece("Heavy", 200)])

    assert worth_showing(shown, ("Play", "Heavy", "Won"), "Play", [piece("Heavy", 500)])
