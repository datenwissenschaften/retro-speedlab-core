from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Self, final

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
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to map its RAM")
        return {
            f.name: (int(f.metadata["address"]), int(f.metadata["length"]))
            for f in fields(cls)
            if "screen" not in f.metadata
        }

    @classmethod
    @final
    def from_ram(cls, raw_ram: Any) -> Self:
        values = {}
        for name, (address, length) in cls.ram_map().items():
            if length == 1:
                values[name] = int(raw_ram[address])
            else:
                values[name] = [int(raw_ram[address + offset]) for offset in range(length)]
        return cls(**values)

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
