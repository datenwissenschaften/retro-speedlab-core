import pytest

from datenwissenschaften.environment.levels import level_map
from datenwissenschaften.states.state import State


class Menu(State):
    pass


class Eat(State):
    pass


class Door(State):
    pass


class Next(State):
    pass


STATES = (Menu, Eat, Door, Next)
LEVELS = (("Level 1", (Eat, Door)),)


def test_a_level_maps_to_its_states():
    assert level_map(LEVELS, STATES) == {"Level 1": ("Eat", "Door")}


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ((("Eat", (Door,)),), "name of a state"),
        ((("Level 1", (Eat, Menu)),), "follow each other"),
        ((("Level 1", ()),), "must be one of"),
    ],
)
def test_levels_must_name_neighbouring_states(levels, message):
    with pytest.raises(ValueError, match=message):
        level_map(levels, STATES)
