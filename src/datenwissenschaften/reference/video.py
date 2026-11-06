from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.reference.samples import SIDE_CROP, small_screen

CHUNK_FRAMES = 2048


def video_chunks(video: Path) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    chunk: list[np.ndarray] = []
    while True:
        read, frame = capture.read()
        if not read:
            break
        chunk.append(small_screen(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), SIDE_CROP))
        if len(chunk) == CHUNK_FRAMES:
            yield np.stack(chunk)
            chunk = []
    capture.release()
    if chunk:
        yield np.stack(chunk)
