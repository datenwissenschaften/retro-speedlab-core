import multiprocessing
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import typer
from loguru import logger

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, RetroSpeedlabPaths, load_config
from datenwissenschaften.tas.feed import SpeedrunFeed
from datenwissenschaften.tas.footage import compare_cut, movie_cut, render_compare, render_movie
from datenwissenschaften.tas.journal import JOURNAL_FILE, JournalStore
from datenwissenschaften.tas.live import LIVE_DIR
from datenwissenschaften.tas.movie import Segment, frames, verified_segments
from datenwissenschaften.tas.paths import PROGRESS_FILE, TAS_MOVIE, tas_dir
from datenwissenschaften.tas.report import course_of
from datenwissenschaften.tas.routes import FULL_RUN, ROUTES_FILE, Leg, Routes, finished, load_progress, parts_of
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.screen import Screen
from datenwissenschaften.tas.search import UNBOUNDED, Crossing
from datenwissenschaften.tas.searcher import Searcher
from datenwissenschaften.tas.stuck import STUCK_FILE, mark_stuck
from datenwissenschaften.ui.relay import relay_key
from datenwissenschaften.ui.video_upload import VideoUpload
from datenwissenschaften.workflow.wrapper import wrapper_class

LOG_FILE = "search.log"
SEARCH_ROUNDS = 2
IMPROVE_ROUNDS = 1

MAIN_WORKER = 0
TRIAL_MOVIE = f"trial-{TAS_MOVIE}"

Footage = Callable[[RetroSpeedlabPaths, Path, tuple[bytes, ...]], Path]

app = typer.Typer()


def segment_of(crossing: Crossing) -> Segment:
    return Segment(crossing.end.state, crossing.pressed.tolist(), crossing.end.emulator_state, crossing.won)


@dataclass(slots=True)
class Tas:
    settings: RetroSpeedlabConfig
    runner: SegmentRunner
    searcher: Searcher
    routes: Routes
    feed: SpeedrunFeed
    movie: Screen
    compare: Screen
    turn: int

    @property
    def output(self) -> Path:
        return tas_dir(self.settings)

    def start(self, leg: Leg, segments: list[Segment], index: int) -> Start:
        if index == 0:
            return leg.entry
        return Start(segments[index - 1].state, segments[index - 1].emulator_state)

    def keep(self, leg: Leg, segments: list[Segment], searching: str) -> list[Segment]:
        self.routes.save(leg, segments)
        (self.output / STUCK_FILE).unlink(missing_ok=True)
        self.feed.show(self.routes.view(leg.name, segments), searching)
        return segments

    def verified(self, leg: Leg, segments: list[Segment]) -> list[Segment]:
        movie = self.output / TAS_MOVIE
        replay = verified_segments(self.runner, leg.entry, segments, movie)
        replayed = [
            replace(segment, emulator_state=end)
            for segment, end in zip(segments[: replay.verified], replay.ends, strict=True)
        ]
        if replay.verified < len(segments):
            logger.warning(f"Only {replay.verified} of {len(segments)} segments of {leg.name} replay; searching again")
        elif frames(replayed):
            self.movie.queue(movie_cut(self.feed.course, leg, replayed, movie, replay.memory))
        return replayed

    def extend(self, leg: Leg, segments: list[Segment]) -> list[Segment]:
        while not finished(segments := self.verified(leg, segments), leg.exit):
            start = self.start(leg, segments, len(segments))
            self.feed.show(self.routes.view(leg.name, segments), start.state)
            crossing = self.searcher.best(start, UNBOUNDED, SEARCH_ROUNDS)
            if crossing is None:
                searches = mark_stuck(self.output / STUCK_FILE, start.state).searches
                self.feed.event("stuck", start.state, None)
                logger.warning(
                    f"No way out of {start.state} after {searches} searches; searching the way into it again"
                )
                segments = segments[:-1]
                continue
            self.feed.event("found", crossing.end.state, crossing.frames)
            segments = self.keep(leg, [*segments, segment_of(crossing)], crossing.end.state)
            logger.info(
                f"{start.state} -> {crossing.end.state} in {crossing.frames} frames; {leg.name} at {frames(segments)}"
            )
            segments = self.shorten(leg, segments, self.turn % len(segments))
            self.turn += 1
        self.feed.show(self.routes.view(leg.name, segments), segments[-1].state)
        return segments

    def improve(self, leg: Leg, segments: list[Segment]) -> list[Segment]:
        for index in range(len(segments)):
            segments = self.shorten(leg, segments, index)
        return segments

    def shorten(self, leg: Leg, segments: list[Segment], index: int) -> list[Segment]:
        start = self.start(leg, segments, index)
        crossing = self.searcher.best(start, len(segments[index].pressed), IMPROVE_ROUNDS)
        if crossing is None or crossing.frames >= len(segments[index].pressed):
            return segments
        saved = len(segments[index].pressed) - crossing.frames
        shorter = [*segments[:index], segment_of(crossing), *segments[index + 1 :]]
        if verified_segments(self.runner, leg.entry, shorter, self.output / TRIAL_MOVIE).verified < len(shorter):
            logger.info(f"Segment {index} of {leg.name} is {saved} frames faster but the rest breaks; it stays")
            return segments
        logger.info(f"Segment {index} of {leg.name} into {crossing.end.state} is {saved} frames faster")
        self.feed.event("improved", crossing.end.state, crossing.frames)
        self.compare.queue(compare_cut(self.runner, self.output, start, segments[index], crossing))
        return self.keep(leg, self.verified(leg, shorter), shorter[-1].state)


def publish(tas: Tas, segments: list[Segment]) -> None:
    tas.movie.settle()
    tas.compare.settle()
    logger.info(f"Verified movie reaches {segments[-1].state} in {frames(segments)} frames")


def video_upload(settings: RetroSpeedlabConfig, name: str) -> VideoUpload | None:
    return VideoUpload(settings.upload.url, relay_key(settings.upload), name) if settings.ui.enabled else None


def screen(settings: RetroSpeedlabConfig, feed: SpeedrunFeed, name: str, render: Footage) -> Screen:
    rendered = partial(render, settings.paths, tas_dir(settings))
    shown = Screen(name, rendered, video_upload(settings, f"{name}.mp4"), partial(feed.shown, name))
    shown.start()
    return shown


def worker_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def tas_run(wrapper: str, config: Path, pool: Executor) -> Tas:
    settings = load_config(config)
    setup_logging(settings.log_level)
    output = tas_dir(settings)
    output.mkdir(parents=True, exist_ok=True)
    logger.add(output / LOG_FILE)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_WORKER))
    course = course_of(runner, settings.speedrun.game)
    journal = JournalStore(output / JOURNAL_FILE)
    feed = SpeedrunFeed(course, journal, output / LIVE_DIR, output / STUCK_FILE, settings.paths.lab_run_marker)
    threading.Thread(target=feed.publish_every, args=(threading.Event(),), name="speedrun-feed", daemon=True).start()
    movie, compare = screen(settings, feed, "movie", render_movie), screen(settings, feed, "compare", render_compare)
    searcher = Searcher(wrapper, config, settings, pool, feed)
    routes = Routes(output / ROUTES_FILE, output / PROGRESS_FILE, parts_of(course))
    return Tas(settings, runner, searcher, routes, feed, movie, compare, 0)


@app.command()
def search(wrapper: str, config: Path, until: str) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        leg = Leg(FULL_RUN, tas.runner.power_on(), until)
        publish(tas, tas.extend(leg, load_progress(tas.routes.progress)))


@app.command()
def verify(wrapper: str, config: Path, routes_file: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_WORKER))
    routes = Routes(
        routes_file, routes_file.with_name(PROGRESS_FILE), parts_of(course_of(runner, settings.speedrun.game))
    )
    power_on = runner.power_on()
    legs = [Leg(part.name, start, part.exit) for part in routes.parts if (start := routes.entry(part, power_on))]
    for leg, segments in [
        *((leg, routes.segments(leg.name)) for leg in legs),
        (Leg(FULL_RUN, power_on, None), load_progress(routes.progress)),
    ]:
        verified = verified_segments(runner, leg.entry, segments, tas_dir(settings) / f"verify-{TAS_MOVIE}").verified
        if verified < len(segments):
            raise RuntimeError(f"Only {verified} of {len(segments)} segments of {leg.name} replay with {wrapper}")
    logger.info(f"Every part of {routes_file} replays with {wrapper}")


if __name__ == "__main__":
    app()
