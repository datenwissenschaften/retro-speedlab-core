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
SPREAD_STEP = 4
MOVING_SHARE = 0.5


@dataclass(slots=True, frozen=True)
class VideoFrames:
    seconds: np.ndarray
    thumbs: np.ndarray


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"{path} is no image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def widest_run(flags: np.ndarray) -> slice:
    edges = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
    starts, ends = edges[::2], edges[1::2]
    widest = int(np.argmax(ends - starts))
    return slice(int(starts[widest]), int(ends[widest]))


def moving_columns(grays: list[np.ndarray]) -> slice:
    spread = np.stack(grays).std(0).mean(0)
    moving = widest_run(spread >= MOVING_SHARE * np.median(spread))
    return slice(moving.start * SPREAD_STEP, moving.stop * SPREAD_STEP)


def video_box(files: list[Path]) -> Box:
    sampled = files[::BOX_SAMPLE_EVERY]
    rows, columns = content_box(reduce(np.maximum, (read_rgb(path) for path in sampled)))
    grays = [read_rgb(path)[rows, columns].mean(2, dtype=np.float32)[::SPREAD_STEP, ::SPREAD_STEP] for path in sampled]
    moving = moving_columns(grays)
    return rows, slice(columns.start + moving.start, min(columns.stop, columns.start + moving.stop))


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
