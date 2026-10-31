from loguru import logger

from datenwissenschaften.reference.paths import TRACE_FILE, reference_dir, route_path, video_path
from datenwissenschaften.reference.samples import collect_samples
from datenwissenschaften.reference.trace import Step, video_chunks, write_route
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import SegmentRunner

SEED = 0


def ensure_guide(settings: RetroSpeedlabConfig, runner: SegmentRunner, route_taken: list[Segment]) -> None:
    route = route_path(settings)
    if not settings.speedrun.guide or route.is_file():
        return
    from datenwissenschaften.reference.model import read_positions, train

    net, known = train(collect_samples(runner, route_taken, SEED))
    steps: list[Step] = []
    first = 0
    for chunk in video_chunks(video_path(settings)):
        reading = read_positions(net, known, chunk)
        for index in range(len(chunk)):
            if reading.playing[index]:
                x, y, height = (int(value) for value in reading.coordinates[index])
                steps.append((first + index, int(reading.areas[index]), x, y, height))
        first += len(chunk)
    write_route(steps, route, reference_dir(settings) / TRACE_FILE)
    logger.info(f"The TAS route has {len(steps)} of {first} video frames in play: {route}")
