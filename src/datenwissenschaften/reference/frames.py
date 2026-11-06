import io
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import cv2
import numpy as np

from datenwissenschaften.reference.thumbnail import Box, content_box, thumbnail
from datenwissenschaften.speedrun.paths import atomic_write

FRAME_SUFFIX = "s.png"
BOX_SAMPLE_EVERY = 10


@dataclass(slots=True, frozen=True)
class VideoFrames:
    seconds: np.ndarray
    thumbs: np.ndarray


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"{path} is no image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def video_box(files: list[Path]) -> Box:
    return content_box(reduce(np.maximum, (read_rgb(path) for path in files[::BOX_SAMPLE_EVERY])))


def read_frames(files: list[Path]) -> VideoFrames:
    if not files:
        raise FileNotFoundError("The reference holds no video frames")
    rows, columns = video_box(files)
    seconds = np.asarray([float(path.name.removesuffix(FRAME_SUFFIX)) for path in files])
    return VideoFrames(seconds, np.stack([thumbnail(read_rgb(path)[rows, columns]) for path in files]))


def fresh(cache: Path, frames: Path) -> bool:
    return cache.is_file() and cache.stat().st_mtime >= frames.stat().st_mtime


def load_frames(frames: Path, cache: Path) -> VideoFrames:
    if fresh(cache, frames):
        with np.load(cache) as stored:
            return VideoFrames(stored["seconds"], stored["thumbs"])
    read = read_frames(sorted(frames.glob(f"*{FRAME_SUFFIX}")))
    buffer = io.BytesIO()
    np.savez(buffer, seconds=read.seconds, thumbs=read.thumbs)
    atomic_write(cache, buffer.getvalue())
    return read
