from dataclasses import dataclass, field, fields, is_dataclass
from functools import cache
from typing import Any, ClassVar, Self, final

import numpy as np
from numpy.typing import NDArray

from datenwissenschaften.route.position import Position


def ram(address: int) -> Any:
    return field(default=0, metadata={"address": address, "length": 1})


def ram_array(address: int, length: int) -> Any:
    if length < 1:
        raise ValueError("length must be positive.")
    return field(default_factory=lambda: [0] * length, metadata={"address": address, "length": length})


def seen(unseen: Any) -> Any:
    return field(default=unseen, metadata={"screen": True})


@dataclass(slots=True, frozen=True)
class Shown:
    label: str
    index: bool

    def value(self, raw: int | None) -> int | None:
        if raw is None or not self.index:
            return raw
        return raw + 1


def viewed(shown: dict[str, Shown], raw: dict[str, int | None]) -> dict[str, int | None]:
    missing = [key for key in shown if key not in raw]
    if missing:
        raise ValueError(f"The game memory has no {missing} to show viewers, only {list(raw)}")
    return {field.label: field.value(raw[key]) for key, field in shown.items()}


@cache
def mapped(cls: type) -> dict[str, tuple[int, int]]:
    name = cls.__name__
    if not is_dataclass(cls):
        raise TypeError(f"{name} must be a dataclass to map its RAM")
    return {
        f.name: (int(f.metadata["address"]), int(f.metadata["length"]))
        for f in fields(cls)
        if "screen" not in f.metadata
    }


@cache
def unread(cls: type) -> dict[str, Any]:
    if hasattr(cls, "__post_init__") or any("__slots__" in vars(base) for base in cls.__mro__[:-1]):
        raise TypeError(f"{cls.__name__} must keep its RAM fields in its __dict__, without slots or __post_init__")
    return {f.name: f.default if "screen" in f.metadata else None for f in fields(cls)}


class RamInfo:
    shown: ClassVar[dict[str, Shown]]

    @classmethod
    @final
    def shown_fields(cls) -> dict[str, Shown]:
        if not hasattr(cls, "shown"):
            raise TypeError(f"{cls.__name__} must declare shown: the game memory viewers see, by label")
        return cls.shown

    @classmethod
    @final
    def ram_map(cls) -> dict[str, tuple[int, int]]:
        return mapped(cls)

    @classmethod
    @final
    def from_ram(cls, raw_ram: bytes | bytearray | NDArray[np.uint8]) -> Self:
        memory = memoryview(raw_ram).cast("B")
        read = {
            name: memory[address] if length == 1 else memory[address : address + length].tolist()
            for name, (address, length) in cls.ram_map().items()
        }
        info = object.__new__(cls)
        vars(info).update(unread(cls))
        vars(info).update(read)
        return info

    def describe(self) -> dict[str, int]:
        return {}

    def position(self) -> Position | None:
        return None

    def remaining_lives(self) -> int | None:
        return None

    def dynamics(self) -> tuple[int, ...]:
        return ()

    def level(self) -> int | None:
        return None

    def progress(self) -> int:
        return 0
