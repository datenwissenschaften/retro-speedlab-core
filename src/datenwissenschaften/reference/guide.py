from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from datenwissenschaften.reference.paths import TRACE_FILE, reference_dir, route_path, video_path
from datenwissenschaften.reference.samples import collect_samples
from datenwissenschaften.reference.trace import Step, rising_levels, video_chunks, write_route
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import SegmentRunner

if TYPE_CHECKING:
    from datenwissenschaften.reference.model import Trained

SEED = 0


def tas_steps(trained: "Trained", settings: RetroSpeedlabConfig) -> tuple[list[Step], int]:
    from datenwissenschaften.reference.model import read_positions

    readings = [read_positions(trained.net, chunk) for chunk in video_chunks(video_path(settings))]
    playing = np.concatenate([reading.playing for reading in readings])
    frames = np.flatnonzero(playing)
    levels = rising_levels(np.concatenate([reading.levels for reading in readings])[playing])
    allowed = trained.allowed[levels]
    areas = np.where(allowed, np.concatenate([reading.areas for reading in readings])[playing], -np.inf).argmax(1)
    coordinates = np.concatenate([reading.coordinates for reading in readings])[playing]
    steps = [
        (int(frame), trained.areas[area], int(x), int(y))
        for frame, area, (x, y), known in zip(frames, areas, coordinates, allowed.any(1), strict=True)
        if known
    ]
    return steps, len(playing)


def ensure_guide(settings: RetroSpeedlabConfig, runner: SegmentRunner, route_taken: list[Segment]) -> None:
    route = route_path(settings)
    if not settings.speedrun.guide or route.is_file():
        return
    from datenwissenschaften.reference.model import train

    steps, frames = tas_steps(train(collect_samples(runner, route_taken, SEED)), settings)
    write_route(steps, route, reference_dir(settings) / TRACE_FILE)
    logger.info(f"The TAS route has {len(steps)} of {frames} video frames in play: {route}")
