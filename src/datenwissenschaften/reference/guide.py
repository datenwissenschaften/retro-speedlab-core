from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from datenwissenschaften.reference.paths import TRACE_FILE, reference_dir, tas_route_path, video_path
from datenwissenschaften.reference.play import Span, dark_screens, level_spans
from datenwissenschaften.reference.samples import Trail, collect_samples
from datenwissenschaften.reference.trace import Timed, timed_rows, video_chunks, write_route
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tas.practice import LevelLoader, level_start
from datenwissenschaften.tas.routes import Routes
from datenwissenschaften.tas.runner import SegmentRunner, Start

if TYPE_CHECKING:
    from datenwissenschaften.reference.model import Trained

SEED = 0


@dataclass(slots=True, frozen=True)
class VideoReading:
    dark: np.ndarray
    places: np.ndarray
    coordinates: np.ndarray
    familiar: np.ndarray


def read_chunk(trained: "Trained", chunk: np.ndarray) -> tuple[np.ndarray, ...]:
    from datenwissenschaften.reference.model import familiar, read_positions

    reading = read_positions(trained.net, chunk)
    return dark_screens(chunk), reading.places, reading.coordinates, familiar(trained, reading)


def read_video(trained: "Trained", settings: RetroSpeedlabConfig) -> VideoReading:
    chunks = [read_chunk(trained, chunk) for chunk in video_chunks(video_path(settings))]
    return VideoReading(*(np.concatenate(parts) for parts in zip(*chunks, strict=True)))


def level_steps(trained: "Trained", video: VideoReading, level: int, span: Span) -> list[Timed]:
    from datenwissenschaften.reference.model import PAGE

    start, end = span
    allowed = trained.allowed[level]
    if not allowed.any():
        return []
    frames = np.arange(start, end)[~video.dark[start:end] & video.familiar[start:end]]
    places = np.where(allowed, video.places[frames], -np.inf).argmax(1)
    return [
        ((int(frame), trained.places[place][0], int(x), trained.places[place][1] * PAGE + int(y)), end)
        for frame, place, (x, y) in zip(frames, places, video.coordinates[frames], strict=True)
    ]


def tas_steps(trained: "Trained", settings: RetroSpeedlabConfig) -> list[list[Timed]]:
    video = read_video(trained, settings)
    spans = level_spans(video.dark, len(trained.allowed))
    for level, (start, end) in enumerate(spans, 1):
        logger.info(f"The TAS plays level {level} in video frames {start}..{end - 1}")
    return [level_steps(trained, video, level, span) for level, span in enumerate(spans)]


def level_starts(runner: SegmentRunner, routes: Routes, loader: LevelLoader | None) -> list[Start] | None:
    later = routes.parts[2:]
    if not later:
        return []
    if loader is None:
        return None
    return [level_start(runner, loader, level, part.entry_state) for level, part in enumerate(later, 1)]


def trails(routes: Routes) -> list[Trail]:
    return [
        (Start(part.entry_state, routes.routes[part.name].entry), routes.segments(part.name))
        for part in routes.parts
        if part.name in routes.routes
    ]


def ensure_guide(
    settings: RetroSpeedlabConfig, runner: SegmentRunner, routes: Routes, loader: LevelLoader | None
) -> None:
    route = tas_route_path(settings)
    if not settings.speedrun.guide or route.is_file():
        return
    starts = level_starts(runner, routes, loader)
    if starts is None:
        logger.info("The TAS guide waits for the level loader: no stored route changes the level yet")
        return
    from datenwissenschaften.reference.model import train

    samples = collect_samples(runner, trails(routes), starts, SEED)
    guided = tas_steps(train(samples, runner.wrapper.level_count), settings)
    missing = [level for level, steps in enumerate(guided, 1) if not steps]
    if missing:
        logger.error(f"The TAS map misses levels {missing}; the search keeps its own route only")
        return
    timed = [step for steps in guided for step in steps]
    write_route([step for step, _ in timed], timed_rows(timed), route, reference_dir(settings) / TRACE_FILE)
    logger.info(f"The TAS route maps {len(timed)} video frames over {len(guided)} levels: {route}")
