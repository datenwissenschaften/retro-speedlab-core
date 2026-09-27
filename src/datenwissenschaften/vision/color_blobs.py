from dataclasses import dataclass

import cv2
import numpy as np

from datenwissenschaften.vision.detection import Detection


@dataclass(slots=True, frozen=True)
class ColorBlobDetector:
    label: str
    colors: tuple[tuple[int, int, int], ...]
    minimum_size: int
    maximum_size: int
    playfield: tuple[int, int, int, int]

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        left, top, right, bottom = self.playfield
        region = frame[top:bottom, left:right]
        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        for color in self.colors:
            mask |= np.all(region == color, axis=2).astype(np.uint8)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        detections = []
        for index in range(1, count):
            x, y, width, height, _ = (int(value) for value in stats[index])
            if self.minimum_size <= max(width, height) <= self.maximum_size:
                detections.append(Detection(self.label, left + x, top + y, width, height))
        return tuple(detections)
