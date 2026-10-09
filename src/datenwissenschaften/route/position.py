from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Position:
    area: int
    x: int
    y: int
    height: int
    airborne: bool
