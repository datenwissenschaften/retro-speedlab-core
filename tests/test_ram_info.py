from dataclasses import dataclass

import numpy as np
import pytest

from datenwissenschaften.ram import RamInfo, ram, ram_array


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


def test_to_dict_reports_scalar_and_list_fields():
    ram_info = _SampleRam(health=10, inventory=[1, 2, 3])

    assert ram_info.to_dict() == {"health": 10, "inventory": [1, 2, 3]}


def test_features_normalizes_scalar_and_list_fields_to_unit_range():
    ram_info = _SampleRam(health=255, inventory=[0, 255, 128])

    features = ram_info.features()

    assert features[0] == pytest.approx(1.0)
    assert features[1:] == [pytest.approx(0.0), pytest.approx(1.0), pytest.approx(128 / 255)]
