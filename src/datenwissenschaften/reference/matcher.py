from dataclasses import dataclass

import numpy as np

MATCH_DISTANCE = 0.2
DISTINCT_MARGIN = 0.1
NEIGHBOUR_CELLS = 1
PLAYFIELD_SHARE = 0.25
FLAT_NORM = 1.0
QUERY_BATCH = 256


@dataclass(slots=True, frozen=True)
class Found:
    frame: int
    screen: int
    distance: float


def playfield_rows(thumbs: np.ndarray) -> np.ndarray:
    spread = thumbs.astype(np.float32).std(0).mean((1, 2))
    return spread >= PLAYFIELD_SHARE * np.median(spread)


def normalized(thumbs: np.ndarray, rows: np.ndarray) -> np.ndarray:
    flat = thumbs[:, rows].reshape(len(thumbs), -1).astype(np.float32)
    centred = flat - flat.mean(1, keepdims=True)
    return centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), FLAT_NORM)


def distinct_second(distances: np.ndarray, cells: np.ndarray, best: np.ndarray) -> np.ndarray:
    chosen = cells[best][:, None]
    apart = np.abs(cells[None, :, 1:] - chosen[:, :, 1:]).max(2) > NEIGHBOUR_CELLS
    distinct = (cells[None, :, 0] != chosen[:, :, 0]) | apart
    return np.where(distinct, distances, np.inf).min(1)


def clear_matches(queries: np.ndarray, screens: np.ndarray, cells: np.ndarray, first: int) -> list[Found]:
    distances = 1 - queries @ screens.T
    best = distances.argmin(1)
    nearest = distances[np.arange(len(best)), best]
    second = distinct_second(distances, cells, best)
    clear = (nearest <= MATCH_DISTANCE) & np.isfinite(second) & (second - nearest >= DISTINCT_MARGIN)
    return [Found(first + int(frame), int(best[frame]), float(nearest[frame])) for frame in np.flatnonzero(clear)]


def match_screens(video: np.ndarray, emulator: np.ndarray, cells: np.ndarray) -> list[Found]:
    if not len(video) or not len(emulator):
        return []
    rows = playfield_rows(emulator)
    queries, screens = normalized(video, rows), normalized(emulator, rows)
    return [
        found
        for first in range(0, len(queries), QUERY_BATCH)
        for found in clear_matches(queries[first : first + QUERY_BATCH], screens, cells, first)
    ]
