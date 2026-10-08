from datenwissenschaften.advisor.backplay import ADVANCE_RATE, SUCCESS_WINDOW, Backplay

ORDER = ("Menu", "Bell", "Door")
POINTS = [b"early", b"middle", b"late"]


def test_practice_starts_at_the_last_point_and_moves_back_after_enough_successes():
    backplay = Backplay({"Bell": POINTS}, ORDER)

    first = {backplay.start("Bell") for _ in range(20)}
    for _ in range(SUCCESS_WINDOW):
        backplay.record("Bell", True)
    widened = {backplay.start("Bell") for _ in range(200)}

    assert first == {b"late"}
    assert widened == {b"middle", b"late"}
    assert backplay.progress("Bell") == "2/3"


def test_failures_keep_the_start_where_it_is():
    backplay = Backplay({"Bell": POINTS}, ORDER)
    successes = int(SUCCESS_WINDOW * ADVANCE_RATE) - 1

    for index in range(SUCCESS_WINDOW):
        backplay.record("Bell", index < successes)

    assert backplay.progress("Bell") == "1/3"


def test_a_finished_or_unknown_state_starts_normally_and_only_later_states_count_as_progress():
    backplay = Backplay({"Bell": [b"only"]}, ORDER)
    for _ in range(SUCCESS_WINDOW):
        backplay.record("Bell", True)

    assert backplay.start("Bell") is None
    assert backplay.start("Menu") is None
    assert backplay.forward("Bell", "Door")
    assert not backplay.forward("Bell", "Menu")
