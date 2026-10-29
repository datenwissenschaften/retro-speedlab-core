import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.ui.media import JPEG_QUALITY, jpeg

TILE_SUFFIX = ".jpg"
MOSAIC_TILES = 9
TILE_SCALE = 0.5


@dataclass(slots=True, frozen=True)
class Mosaic:
    jpeg: bytes
    seeds: list[int]


def tile(screen: np.ndarray) -> bytes:
    return jpeg(cv2.resize(screen, None, fx=TILE_SCALE, fy=TILE_SCALE, interpolation=cv2.INTER_AREA))


def shown_tiles(directory: Path) -> list[tuple[int, bytes]]:
    found = []
    worker_tiles = (path for path in directory.glob(f"*{TILE_SUFFIX}") if path.stem.isdigit())
    for seed, path in sorted((int(path.stem), path) for path in worker_tiles):
        try:
            found.append((seed, path.read_bytes()))
        except FileNotFoundError:
            continue
    return found[:MOSAIC_TILES]


def compose(directory: Path) -> Mosaic | None:
    tiles = shown_tiles(directory)
    if not tiles:
        return None
    pictures = [cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) for _, body in tiles]
    columns = math.ceil(math.sqrt(len(pictures)))
    height, width, channels = pictures[0].shape
    canvas = np.zeros((math.ceil(len(pictures) / columns) * height, columns * width, channels), np.uint8)
    for index, picture in enumerate(pictures):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = picture
    encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
    return Mosaic(encoded, [seed for seed, _ in tiles])
