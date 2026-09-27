from pathlib import Path

import cv2
import numpy as np
import pytest

from datenwissenschaften.vision.color_blobs import ColorBlobDetector
from datenwissenschaften.vision.detection import Detection
from datenwissenschaften.vision.overlay import draw_detections
from datenwissenschaften.vision.relative_position import describe_target, nearest_distance
from datenwissenschaften.vision.template_detector import TemplateDetector

SNAKE = (248, 116, 96)


def _pattern() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 255, (12, 10, 3), dtype=np.uint8)


def _template(tmp_path: Path) -> Path:
    path = tmp_path / "door.png"
    cv2.imwrite(str(path), cv2.cvtColor(_pattern(), cv2.COLOR_RGB2BGR))
    return path


def test_template_detector_finds_every_separate_match(tmp_path: Path):
    frame = np.zeros((80, 100, 3), np.uint8)
    frame[10:22, 20:30] = _pattern()
    frame[50:62, 70:80] = _pattern()
    detector = TemplateDetector("door", (_template(tmp_path),), 0.9, 4.0)

    detections = detector.detect(frame)

    assert sorted((d.left, d.top) for d in detections) == [(20, 10), (70, 50)]
    assert {d.label for d in detections} == {"door"}
    assert detections[0].center == (detections[0].left + 5, detections[0].top + 6)


def test_template_detector_needs_templates_that_exist(tmp_path: Path):
    with pytest.raises(ValueError):
        TemplateDetector("door", (), 0.9, 4.0)
    with pytest.raises(FileNotFoundError):
        TemplateDetector("door", (tmp_path / "missing.png",), 0.9, 4.0)


def test_color_blobs_are_filtered_by_size_and_playfield():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[20:36, 30:46] = SNAKE
    frame[60:64, 60:64] = SNAKE
    frame[92:99, 5:20] = SNAKE
    detector = ColorBlobDetector("snake", (SNAKE,), 10, 20, (0, 0, 100, 90))

    assert detector.detect(frame) == (Detection("snake", 30, 20, 16, 16),)


def test_targets_are_described_relative_to_the_actor():
    actor = Detection("snake", 50, 50, 10, 10)
    far = Detection("door", 90, 90, 10, 10)
    near = Detection("door", 10, 20, 10, 10)

    assert describe_target(actor, (far, near)) == {"visible": True, "direction": "up-left", "distance": 50}
    assert describe_target(actor, (Detection("door", 52, 30, 10, 10),))["direction"] == "up"
    assert describe_target(actor, (Detection("door", 51, 51, 10, 10),))["direction"] == "here"
    assert nearest_distance(actor, (far, near)) == pytest.approx(50.0)


def test_unseen_targets_or_actor_are_reported_honestly():
    target = Detection("door", 0, 0, 4, 4)

    assert describe_target(None, (target,)) == {"visible": True, "direction": "unknown", "distance": None}
    assert describe_target(Detection("snake", 0, 0, 4, 4), ()) == {
        "visible": False,
        "direction": "unknown",
        "distance": None,
    }
    assert nearest_distance(None, (target,)) is None


def test_overlay_draws_boxes_without_touching_the_original():
    frame = np.zeros((40, 40, 3), np.uint8)

    annotated = draw_detections(frame, (Detection("door", 10, 10, 8, 8),))

    assert frame.sum() == 0
    assert annotated[10, 10].tolist() == [255, 255, 0]
