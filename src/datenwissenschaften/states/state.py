from abc import ABC
from typing import Any, Generic, TypeVar

import numpy as np

from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.vision.detection import Detection

T = TypeVar("T", bound=RamInfo)


class State(ABC, Generic[T]):
    description: str

    ram: T
    frame: np.ndarray

    def __init__(self, landmarks: Landmarks) -> None:
        self.landmarks = landmarks

    def reset(self, ram: T, frame: np.ndarray) -> None:
        self.ram = ram
        self.frame = frame
        self._see()
        self._on_reset()

    def step(self, ram: T, frame: np.ndarray) -> tuple[float, bool, bool, type["State[T]"] | None]:
        self.ram = ram
        self.frame = frame
        self._see()
        return self._reward(), self._terminated(), self._truncated(), self._next()

    def describe(self) -> dict[str, Any]:
        return {}

    def detections(self) -> tuple[Detection, ...]:
        return ()

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
