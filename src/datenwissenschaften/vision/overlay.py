import cv2
import numpy as np

from datenwissenschaften.vision.detection import Detection

BOX_COLOR = (255, 255, 0)
TEXT_COLOR = (255, 255, 255)
FONT_SCALE = 0.3
LABEL_OFFSET = 3
MINIMUM_LABEL_TOP = 8


def draw_detections(frame: np.ndarray, detections: tuple[Detection, ...]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        corner = (detection.left + detection.width, detection.top + detection.height)
        cv2.rectangle(annotated, (detection.left, detection.top), corner, BOX_COLOR, 1)
        label_position = (detection.left, max(detection.top - LABEL_OFFSET, MINIMUM_LABEL_TOP))
        cv2.putText(annotated, detection.label, label_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOR, 1)
    return annotated
