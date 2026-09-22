from datenwissenschaften.helpers.position import Position


def test_coordinates_combine_screen_and_local_position():
    position = Position(10, 20, screen_x=1, screen_y=2, screen_size=100)

    assert position.x == 110
    assert position.y == 220
    assert position.coordinates == (110, 220)
    assert position.screen == (1, 2)


def test_viewport_strips_the_screen_offset():
    position = Position(10, 20, screen_x=1, screen_y=2, screen_size=100)

    viewport = position.viewport

    assert viewport.coordinates == (10, 20)
    assert viewport.screen == (0, 0)


def test_on_screen_returns_a_new_position_with_the_given_screen():
    position = Position(10, 20, screen_size=100)

    relocated = position.on_screen((3, 4))

    assert relocated.screen == (3, 4)
    assert relocated.position_x == 10
    assert relocated.position_y == 20


def test_is_zero_detects_the_origin_position():
    assert Position(0, 0).is_zero is True
    assert Position(1, 0).is_zero is False


def test_distance_to_a_missing_or_zero_position_is_zero():
    position = Position(10, 10)

    assert position.distance_to(None) == 0.0
    assert position.distance_to(Position(0, 0)) == 0.0


def test_distance_to_computes_euclidean_distance():
    origin = Position(0, 0)
    target = Position(3, 4)

    assert origin.distance_to(target) == 5.0


def test_speed_to_a_missing_or_zero_position_is_zero():
    position = Position(10, 10)

    assert position.speed_to(None) == 0.0
    assert position.speed_to(Position(0, 0)) == 0.0


def test_speed_to_divides_distance_by_elapsed_time():
    previous = Position(1, 1)
    current = Position(4, 5)

    assert current.speed_to(previous, dt=2.0) == 2.5


def test_subtracting_a_missing_or_zero_position_returns_zero_offset():
    position = Position(10, 10)

    assert (position - None) == (0, 0)
    assert (position - Position(0, 0)) == (0, 0)


def test_subtracting_a_valid_position_returns_the_pixel_offset():
    a = Position(10, 20)
    b = Position(3, 4)

    assert (a - b) == (7, 16)
