import cv2
import numpy as np

from datenwissenschaften.reference.play import BLACK

THUMB_WIDTH = 32
THUMB_HEIGHT = 30
THUMB_SHAPE = (THUMB_HEIGHT, THUMB_WIDTH, 3)

Box = tuple[slice, slice]


def thumbnail(screen: np.ndarray) -> np.ndarray:
    return cv2.resize(screen, (THUMB_WIDTH, THUMB_HEIGHT), interpolation=cv2.INTER_AREA)


def content_box(brightest: np.ndarray) -> Box:
    lit = brightest.max(2) >= BLACK
    rows, columns = np.flatnonzero(lit.any(1)), np.flatnonzero(lit.any(0))
    if not rows.size:
        raise ValueError("The reference video shows nothing but black")
    return slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1)
