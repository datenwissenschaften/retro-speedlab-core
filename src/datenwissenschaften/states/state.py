from abc import ABC
from typing import Generic, TypeVar

from datenwissenschaften.ram import RamInfo

T = TypeVar("T", bound=RamInfo)


class State(ABC, Generic[T]):
    goal: str
    ram: T

    def reset(self, ram: T) -> None:
        self.ram = ram
        self._see()
        self._on_reset()

    def step(self, ram: T) -> tuple[float, bool, bool, type["State[T]"] | None]:
        self.ram = ram
        self._see()
        return self._reward(), self._terminated(), self._truncated(), self._next()

    def _see(self) -> None:
        pass

    def _on_reset(self) -> None:
        pass

    def _reward(self) -> float:
        return 0.0

    def _terminated(self) -> bool:
        return False

    def _truncated(self) -> bool:
        return False

    def _won(self) -> bool:
        return False

    def _next(self) -> type["State[T]"] | None:
        return None
