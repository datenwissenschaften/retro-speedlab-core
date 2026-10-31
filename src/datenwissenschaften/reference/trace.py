import json
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.reference.samples import SIDE_CROP, small_screen
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import Cell, cell_of
from datenwissenschaften.tas.paths import atomic_write

CHUNK_FRAMES = 2048
LEVEL_GAP_FRAMES = 60

Step = tuple[int, int, int, int]


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


def rising_levels(log_probs: np.ndarray) -> np.ndarray:
    levels = np.arange(log_probs.shape[1])
    best, back = log_probs[0].copy(), np.zeros(log_probs.shape, dtype=int)
    for frame in range(1, len(log_probs)):
        prefix = np.maximum.accumulate(best)
        back[frame] = np.maximum.accumulate(np.where(best >= prefix, levels, 0))
        best = prefix + log_probs[frame]
    path = np.empty(len(log_probs), dtype=int)
    path[-1] = int(np.argmax(best))
    for frame in range(len(log_probs) - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


def stretches(steps: list[Step]) -> list[list[Step]]:
    found: list[list[Step]] = []
    for step in steps:
        if found and step[0] - found[-1][-1][0] <= LEVEL_GAP_FRAMES:
            found[-1].append(step)
        else:
            found.append([step])
    return found


def route_rows(steps: list[Step]) -> list[list[int]]:
    remaining: dict[Cell, tuple[int, int]] = {}
    for stretch in stretches(steps):
        end = stretch[-1][0]
        for frame, area, x, y in stretch:
            cell = cell_of(Position(area, x, y, 0, False))
            if cell not in remaining or end - frame < remaining[cell][1]:
                remaining[cell] = (frame, end - frame)
    return [[*cell, frame, left] for cell, (frame, left) in sorted(remaining.items())]


def write_route(steps: list[Step], route: Path, trace: Path) -> None:
    atomic_write(trace, json.dumps(steps).encode())
    atomic_write(route, json.dumps(route_rows(steps)).encode())
