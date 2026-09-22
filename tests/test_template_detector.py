from pathlib import Path

import cv2
import numpy as np
import pytest

from datenwissenschaften.helpers.position import Position
from datenwissenschaften.vision.template_detector import TemplateDetector


def _template_array(size: int = 8) -> np.ndarray:
    pattern = np.zeros((size, size), dtype=np.uint8)
    pattern[::2, ::2] = 255
    pattern[1::2, 1::2] = 255
    return pattern


def _write_template(path: Path, *, size: int = 8) -> None:
    cv2.imwrite(str(path), _template_array(size))


def _noise_frame(size: int = 32) -> np.ndarray:
    return np.random.default_rng(seed=0).integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _frame_with_template(template_size: int, frame_size: int, offset: int) -> np.ndarray:
    frame = _noise_frame(frame_size)
    template = _template_array(template_size)
    for channel in range(3):
        frame[offset : offset + template_size, offset : offset + template_size, channel] = template
    return frame


def test_detector_resolves_an_absolute_template_path(tmp_path: Path):
    template_path = tmp_path / "target.png"
    _write_template(template_path)

    detector = TemplateDetector(template_path)

    assert detector.template_path == str(template_path.resolve())
    assert detector.template_h == 8
    assert detector.template_w == 8


def test_detector_resolves_a_relative_path_under_assets(tmp_path: Path, monkeypatch):
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    _write_template(assets_dir / "target.png")
    monkeypatch.chdir(tmp_path)

    detector = TemplateDetector("target.png")

    assert detector.template_path == str((assets_dir / "target.png").resolve())


def test_detector_raises_file_not_found_for_a_missing_template(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        TemplateDetector(tmp_path / "missing.png")


def test_detect_reports_no_match_below_threshold(tmp_path: Path):
    template_path = tmp_path / "target.png"
    _write_template(template_path)
    detector = TemplateDetector(template_path, threshold=0.85)

    detector.detect(_noise_frame())

    assert detector.seen is False
    assert detector.position is None
    assert detector.score is None
    assert detector.distance(Position(0, 0)) is None


def test_detect_reports_a_match_above_threshold_for_a_color_frame(tmp_path: Path):
    template_path = tmp_path / "target.png"
    _write_template(template_path)
    detector = TemplateDetector(template_path, threshold=0.85)

    frame = _frame_with_template(template_size=8, frame_size=32, offset=10)
    detector.detect(frame)

    assert detector.seen is True
    assert detector.score == pytest.approx(1.0)
    assert detector.position == Position(position_x=14, position_y=14)
    assert detector.distance(detector.position) == 0.0


def test_detect_accepts_a_grayscale_frame(tmp_path: Path):
    template_path = tmp_path / "target.png"
    _write_template(template_path)
    detector = TemplateDetector(template_path, threshold=0.85)

    frame = cv2.cvtColor(_frame_with_template(template_size=8, frame_size=32, offset=10), cv2.COLOR_RGB2GRAY)
    detector.detect(frame)

    assert detector.seen is True
