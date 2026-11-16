from speedlab_engine.cartridge import BUTTONS
from speedlab_engine.collection import DEMO_COLLECTION, cartridge_name
from speedlab_engine.pad import packed

from datenwissenschaften.speedrun.level_clock import clocked

RIGHT = bytes([packed(tuple(name == "RIGHT" for name in BUTTONS))])
WON_AT = 184


def test_a_won_run_is_clocked_at_every_level_entry_and_its_win():
    clock = clocked(DEMO_COLLECTION, cartridge_name("walk"), RIGHT * (WON_AT + 1))

    assert clock == {
        "cartridge": "games.walk.cartridge",
        "version": "1.0.0",
        "levels": [[0, 0], [48, 1], [112, 2], [176, 3]],
        "won_at": WON_AT,
    }


def test_an_unfinished_run_has_no_win():
    assert clocked(DEMO_COLLECTION, cartridge_name("walk"), RIGHT)["won_at"] is None
