from dataclasses import dataclass

import numpy as np
import pytest

from datenwissenschaften.ram import RamInfo, Shown, ram, ram_array, seen, viewed


@dataclass
class _SampleRam(RamInfo):
    health: int = ram(0x10)
    inventory: list = ram_array(0x20, 3)


def test_ram_array_rejects_a_non_positive_length():
    with pytest.raises(ValueError, match="length must be positive"):
        ram_array(0x00, 0)


def test_ram_map_reports_addresses_and_lengths_from_field_metadata():
    assert _SampleRam.ram_map() == {"health": (0x10, 1), "inventory": (0x20, 3)}


def test_from_ram_reads_scalar_and_array_fields_from_raw_memory():
    raw = np.zeros(64, dtype=np.uint8)
    raw[0x10] = 42
    raw[0x20:0x23] = [1, 2, 3]

    ram_info = _SampleRam.from_ram(raw)

    assert ram_info.health == 42
    assert ram_info.inventory == [1, 2, 3]


def test_a_field_seen_on_the_screen_has_no_address_and_starts_unseen():
    @dataclass(frozen=True)
    class Screened(RamInfo):
        lives: int = ram(0x10)
        on_map: bool = seen(False)
        health: int | None = seen(None)

    assert Screened.ram_map() == {"lives": (0x10, 1)}
    screened = Screened.from_ram(bytes(32))
    assert screened.on_map is False and screened.health is None


def test_a_game_without_known_ram_meanings_reports_none():
    @dataclass(frozen=True)
    class Unknown(RamInfo):
        pass

    ram_info = Unknown.from_ram(bytes(16))

    assert (ram_info.position(), ram_info.remaining_lives(), ram_info.dynamics()) == (None, None, ())


def test_shown_fields_name_the_memory_viewers_see_and_fail_when_undeclared():
    @dataclass(frozen=True)
    class Declared(RamInfo):
        shown = {"level": Shown("Level", True)}

    with pytest.raises(ValueError, match="no \['level'\] to show viewers"):
        viewed(Declared.shown_fields(), {"lives": 3})
    with pytest.raises(TypeError, match="_SampleRam must declare shown"):
        _SampleRam.shown_fields()
    assert viewed(Declared.shown_fields(), {"level": 0}) == {"Level": 1}


def test_from_ram_builds_the_same_ram_info_as_its_dataclass_init():
    @dataclass(frozen=True)
    class Mixed(RamInfo):
        lives: int = ram(0x02)
        on_map: bool = seen(False)
        tiles: list[int] = ram_array(0x04, 4)
        mode: int = ram(0x01)

    raw = np.arange(16, dtype=np.uint8)
    read = Mixed.from_ram(raw)

    assert read == Mixed(lives=2, tiles=[4, 5, 6, 7], mode=1)
    assert list(vars(read).items()) == list(vars(Mixed(lives=2, tiles=[4, 5, 6, 7], mode=1)).items())
    assert Mixed.from_ram(bytes(raw)) == Mixed.from_ram(bytearray(raw)) == read


def test_from_ram_rejects_ram_info_it_cannot_fill_without_its_init():
    @dataclass(frozen=True, slots=True)
    class Slotted(RamInfo):
        lives: int = ram(0x02)

    @dataclass
    class Checked(RamInfo):
        lives: int = ram(0x02)

        def __post_init__(self) -> None:
            self.lives = 0

    for ram_info in (Slotted, Checked):
        with pytest.raises(TypeError, match="without slots or __post_init__"):
            ram_info.from_ram(bytes(4))
