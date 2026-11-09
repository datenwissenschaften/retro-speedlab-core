import json
from concurrent.futures import Executor, Future
from pathlib import Path

import msgspec
from loguru import logger

from datenwissenschaften.reference.fetch import FRAMES_PER_SECOND
from datenwissenschaften.reference.frames import load_frames
from datenwissenschaften.reference.matches import LevelMatches, LevelReading, Matches, read_level
from datenwissenschaften.reference.paths import FRAMES_DIR, MATCHES_FILE, SCREENS_FILE, THUMBS_FILE, reference_dir
from datenwissenschaften.reference.play import UnsplitVideo, dark_screens, level_spans
from datenwissenschaften.reference.screens import updated_screens
from datenwissenschaften.route.paths import video_route_path
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.paths import archives_dir, atomic_write
from datenwissenschaften.speedrun.runner import SegmentRunner
from datenwissenschaften.speedrun.workers import worker_runner


def read_reference(runner: SegmentRunner, settings: RetroSpeedlabConfig) -> list[LevelReading]:
    directory = reference_dir(settings)
    video = load_frames(directory / FRAMES_DIR, directory / THUMBS_FILE)
    shots = updated_screens(runner, archives_dir(settings), directory / SCREENS_FILE)
    dark = dark_screens(video.thumbs)
    spans = level_spans(dark, runner.wrapper.level_count, FRAMES_PER_SECOND)
    rate = runner.emulator.em.get_screen_rate()
    return [read_level(video, ~dark, span, level, shots, rate) for level, span in enumerate(spans)]


def write_matches(settings: RetroSpeedlabConfig, readings: list[LevelReading]) -> None:
    matches = [match for reading in readings for match in reading.matches]
    rows = [row for reading in readings for row in reading.rows]
    found = Matches([reading.summary for reading in readings], matches)
    atomic_write(reference_dir(settings) / MATCHES_FILE, msgspec.json.encode(found))
    atomic_write(video_route_path(settings), json.dumps(rows).encode())


def match_reference(wrapper: str, config: Path) -> list[LevelMatches]:
    settings = load_config(config)
    readings = read_reference(worker_runner(wrapper, config), settings)
    write_matches(settings, readings)
    return [reading.summary for reading in readings]


def progress(levels: list[LevelMatches]) -> list[dict[str, int]]:
    return [{"level": level.level, "matched": level.matched, "frames": level.frames} for level in levels]


class ReferenceMatching:
    def __init__(self, wrapper: str, config: Path, pool: Executor, feed: SpeedrunFeed) -> None:
        self.wrapper = wrapper
        self.config = config
        self.frames = reference_dir(load_config(config)) / FRAMES_DIR
        self.pool = pool
        self.feed = feed
        self.running: Future[list[LevelMatches]] | None = None
        self.unsplit = False

    def refresh(self) -> None:
        if self.unsplit or not self.frames.is_dir() or (self.running is not None and not self.running.done()):
            return
        self.running = self.pool.submit(match_reference, self.wrapper, self.config)
        self.running.add_done_callback(self.matched)

    def matched(self, done: Future[list[LevelMatches]]) -> None:
        error = done.exception()
        if isinstance(error, UnsplitVideo):
            logger.warning(f"{error}; the reference video gives this game no route map")
            self.unsplit = True
            return
        if error is not None:
            logger.opt(exception=error).error("The reference video could not be matched with emulator screens")
            return
        levels = done.result()
        for level in levels:
            logger.info(
                f"Level {level.level}: {level.matched} of {level.frames} video frames match an emulator screen, "
                f"{len(level.waypoints)} waypoints"
            )
        self.feed.referenced(progress(levels))
