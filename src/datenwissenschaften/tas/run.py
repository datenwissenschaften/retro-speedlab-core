import multiprocessing
import random
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import msgspec
import typer
from loguru import logger

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.wrapper import phases
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, RetroSpeedlabPaths, load_config
from datenwissenschaften.tas.feed import SpeedrunFeed
from datenwissenschaften.tas.footage import compare_cut, movie_cut, render_compare, render_movie
from datenwissenschaften.tas.journal import JOURNAL_FILE, JournalStore
from datenwissenschaften.tas.live import LIVE_DIR
from datenwissenschaften.tas.live_search import search_round
from datenwissenschaften.tas.movie import Segment, frames, verified_segments
from datenwissenschaften.tas.paths import PROGRESS_FILE, TAS_MOVIE, atomic_write, tas_dir
from datenwissenschaften.tas.report import course_of
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.screen import Screen
from datenwissenschaften.tas.search import UNBOUNDED, Crossing, fastest_distinct
from datenwissenschaften.tas.stuck import STUCK_FILE, mark_stuck
from datenwissenschaften.tas.workers import explore, polish
from datenwissenschaften.ui.relay import relay_key
from datenwissenschaften.ui.video_upload import VideoUpload
from datenwissenschaften.workflow.wrapper import wrapper_class

LOG_FILE = "search.log"
SEARCH_ROUNDS = 2
IMPROVE_ROUNDS = 1

SEED_SPACE = 2**31
MAIN_WORKER = 0

Footage = Callable[[RetroSpeedlabPaths, Path, tuple[bytes, ...]], Path]

app = typer.Typer()


def load_progress(path: Path) -> list[Segment]:
    return msgspec.msgpack.decode(path.read_bytes(), type=list[Segment]) if path.is_file() else []


def save_progress(path: Path, segments: list[Segment]) -> None:
    atomic_write(path, msgspec.msgpack.encode(segments))


def finished(segments: list[Segment], until: str) -> bool:
    return bool(segments) and (segments[-1].won or segments[-1].state == until)


def segment_of(crossing: Crossing) -> Segment:
    return Segment(crossing.end.state, crossing.pressed.tolist(), crossing.end.emulator_state, crossing.won)


@dataclass(slots=True)
class Tas:
    wrapper: str
    config: Path
    settings: RetroSpeedlabConfig
    runner: SegmentRunner
    pool: Executor
    feed: SpeedrunFeed
    movie: Screen
    compare: Screen
    turn: int

    @property
    def output(self) -> Path:
        return tas_dir(self.settings)

    def start(self, segments: list[Segment], index: int) -> Start:
        if index == 0:
            return self.runner.power_on()
        return Start(segments[index - 1].state, segments[index - 1].emulator_state)

    def candidates(self, start: Start, bound: int, rounds: int) -> list[Crossing]:
        for round_index in range(rounds):
            seconds = self.settings.speedrun.segment_seconds * 2**round_index
            first = random.randrange(SEED_SPACE)
            self.feed.begin(search_round(start.state, round_index, rounds, seconds))
            searches = [
                self.pool.submit(explore, *self._job(start, bound, first + index, seconds), self.feed.live)
                for index in range(self.settings.speedrun.workers)
            ]
            found = fastest_distinct([crossing for search in searches for crossing in search.result()])
            self.feed.end()
            if found:
                return found
        return []

    def best(self, start: Start, bound: int, rounds: int) -> Crossing | None:
        found = self.candidates(start, bound, rounds)
        if not found:
            return None
        seconds = self.settings.speedrun.segment_seconds
        ahead = [
            self.pool.submit(explore, *self._job(crossing.end, UNBOUNDED, seed, seconds), None)
            for seed, crossing in enumerate(found)
        ]
        following = [min((after.frames for after in search.result()), default=UNBOUNDED) for search in ahead]
        totals = [
            crossing.frames + (0 if crossing.won else after) for crossing, after in zip(found, following, strict=True)
        ]
        chosen = found[totals.index(min(totals))]
        return self.pool.submit(polish, self.wrapper, self.config, start, chosen, seconds).result()

    def keep(self, segments: list[Segment], searching: str) -> list[Segment]:
        save_progress(self.output / PROGRESS_FILE, segments)
        (self.output / STUCK_FILE).unlink(missing_ok=True)
        self.feed.show(segments, searching)
        return segments

    def verified(self, segments: list[Segment]) -> list[Segment]:
        movie = self.output / TAS_MOVIE
        replay = verified_segments(self.runner, segments, movie)
        if replay.verified == len(segments):
            if frames(segments):
                self.movie.queue(movie_cut(self.feed.course, segments, movie, replay.memory))
            return segments
        logger.warning(f"Only {replay.verified} of {len(segments)} segments replay from power-on; searching again")
        return segments[: replay.verified]

    def extend(self, segments: list[Segment], until: str) -> list[Segment]:
        while not finished(segments := self.verified(segments), until):
            start = self.start(segments, len(segments))
            self.feed.show(segments, start.state)
            crossing = self.best(start, UNBOUNDED, SEARCH_ROUNDS)
            if crossing is None:
                searches = mark_stuck(self.output / STUCK_FILE, start.state).searches
                self.feed.event("stuck", start.state, None)
                logger.warning(
                    f"No way out of {start.state} after {searches} searches; searching the way into it again"
                )
                segments = segments[:-1]
                continue
            self.feed.event("found", crossing.end.state, crossing.frames)
            segments = self.keep([*segments, segment_of(crossing)], crossing.end.state)
            logger.info(f"{start.state} -> {crossing.end.state} in {crossing.frames} frames; run at {frames(segments)}")
            segments = self.shorten(segments, self.turn % len(segments))
            self.turn += 1
        self.feed.show(segments, segments[-1].state)
        return segments

    def improve(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            raise RuntimeError(f"{self.output / PROGRESS_FILE} holds no movie to improve; search one first")
        until, index = segments[-1].state, 0
        while index < len(segments):
            if (shorter := self.shorten(segments, index)) is not segments:
                segments = self.extend(shorter, until)
            index += 1
        return segments

    def shorten(self, segments: list[Segment], index: int) -> list[Segment]:
        start = self.start(segments, index)
        crossing = self.best(start, len(segments[index].pressed), IMPROVE_ROUNDS)
        if crossing is None or crossing.frames >= len(segments[index].pressed):
            return segments
        saved = len(segments[index].pressed) - crossing.frames
        logger.info(f"Segment {index} into {crossing.end.state} is {saved} frames faster; the rest is rebuilt")
        self.feed.event("improved", crossing.end.state, crossing.frames)
        self.compare.queue(compare_cut(self.runner, self.output, start, segments[index], crossing))
        return self.keep([*segments[:index], segment_of(crossing)], start.state)

    def _job(self, start: Start, bound: int, seed: int, seconds: float) -> tuple[str, Path, Start, int, int, float]:
        return self.wrapper, self.config, start, bound, seed, seconds


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
    tas_dir(settings).mkdir(parents=True, exist_ok=True)
    logger.add(tas_dir(settings) / LOG_FILE)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_WORKER))
    output = tas_dir(settings)
    course = course_of(runner, settings.speedrun.game)
    journal = JournalStore(output / JOURNAL_FILE)
    feed = SpeedrunFeed(course, journal, output / LIVE_DIR, output / STUCK_FILE, settings.paths.lab_run_marker)
    threading.Thread(target=feed.publish_every, args=(threading.Event(),), name="speedrun-feed", daemon=True).start()
    movie, compare = screen(settings, feed, "movie", render_movie), screen(settings, feed, "compare", render_compare)
    return Tas(wrapper, config, settings, runner, pool, feed, movie, compare, 0)


def speedrun_forever(wrapper: str, config: Path) -> None:
    with worker_pool(load_config(config).speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        progress = tas.output / PROGRESS_FILE
        if not progress.is_file():
            save_progress(progress, [])
        segments = tas.extend(load_progress(progress), phases(wrapper_class(wrapper))[-1])
        while True:
            segments = tas.improve(segments)


@app.command()
def search(wrapper: str, config: Path, until: str) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        segments = tas.extend(load_progress(tas.output / PROGRESS_FILE), until)
        publish(tas, segments)


@app.command()
def improve(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        segments = tas.improve(tas.verified(load_progress(tas.output / PROGRESS_FILE)))
        publish(tas, segments)


@app.command()
def verify(wrapper: str, config: Path, progress: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    segments = load_progress(progress)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_WORKER))
    verified = verified_segments(runner, segments, tas_dir(settings) / f"verify-{TAS_MOVIE}").verified
    if verified < len(segments):
        raise RuntimeError(f"Only {verified} of {len(segments)} segments of {progress} replay with {wrapper}")
    logger.info(f"All {len(segments)} segments of {progress} replay with {wrapper}")


if __name__ == "__main__":
    app()
