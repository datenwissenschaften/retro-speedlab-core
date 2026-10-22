from datenwissenschaften.environment.stillness import Stillness
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.state_route import CELL_SIZE

MAX_STILL_FRAMES = 3


def at(x: int) -> Position:
    return Position(area=0, x=x, y=0, height=0, airborne=False)


def test_an_attempt_stalls_after_too_long_without_a_new_spot():
    stillness = Stillness(MAX_STILL_FRAMES)

    stalls = [stillness.stalled(at(0)) for _ in range(MAX_STILL_FRAMES + 1)]

    assert stalls == [False, False, False, True]


def test_pacing_between_known_spots_still_stalls():
    stillness = Stillness(MAX_STILL_FRAMES)
    stillness.stalled(at(0))
    stillness.stalled(at(CELL_SIZE))

    stalls = [stillness.stalled(at(CELL_SIZE * (frame % 2))) for frame in range(MAX_STILL_FRAMES)]

    assert stalls == [False, False, True]


def test_a_new_spot_and_a_restart_reset_the_clock():
    stillness = Stillness(MAX_STILL_FRAMES)
    for _ in range(MAX_STILL_FRAMES - 1):
        stillness.stalled(at(0))

    moved = stillness.stalled(at(CELL_SIZE))
    stillness.restart()

    assert moved is False
    assert (stillness.visited, stillness.still_frames) == (set(), 0)


def test_frames_without_a_position_never_stall():
    stillness = Stillness(MAX_STILL_FRAMES)

    assert not any(stillness.stalled(None) for _ in range(MAX_STILL_FRAMES * 2))
