import zlib

import numpy as np

from datenwissenschaften.states.facts import Facts, Offset

FACT_SLOTS = 64
RAM_SCALE = 255.0


def encode(ram: np.ndarray, facts: Facts) -> np.ndarray:
    slots = np.zeros(FACT_SLOTS, dtype=np.float32)
    for name, value in facts.items():
        for key, number in _numbers(name, value):
            slots[zlib.crc32(key.encode()) % FACT_SLOTS] += number
    return np.concatenate([ram.astype(np.float32) / RAM_SCALE, slots])


def _numbers(name: str, value: object) -> list[tuple[str, float]]:
    if isinstance(value, Offset):
        return [(f"{name}.right", float(value.right)), (f"{name}.down", float(value.down))]
    if isinstance(value, bool | int | float):
        return [(name, float(value))]
    return [(f"{name}={value}", 1.0)]
