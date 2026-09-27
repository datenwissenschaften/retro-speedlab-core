from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.vision.detection import Detection


class TemplateDetector:
    def __init__(self, label: str, template_paths: tuple[Path, ...], threshold: float, minimum_distance: float) -> None:
        if not template_paths:
            raise ValueError("At least one template is required.")
        self.label = label
        self.templates = tuple(self._load(path) for path in template_paths)
        self.threshold = threshold
        self.minimum_distance = minimum_distance

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        candidates: list[tuple[float, Detection]] = []
        for template in self.templates:
            height, width = template.shape
            scores = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
            for top, left in np.argwhere(scores >= self.threshold):
                candidates.append((float(scores[top, left]), Detection(self.label, int(left), int(top), width, height)))
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        selected: list[Detection] = []
        for _, detection in candidates:
            if all(self._apart(detection, other) for other in selected):
                selected.append(detection)
        return tuple(selected)

    def _apart(self, first: Detection, second: Detection) -> bool:
        (first_x, first_y), (second_x, second_y) = first.center, second.center
        return float(np.hypot(first_x - second_x, first_y - second_y)) >= self.minimum_distance

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        template = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            raise FileNotFoundError(path)
        return template
