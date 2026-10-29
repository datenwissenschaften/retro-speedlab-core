import multiprocessing
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import msgspec
import typer
from loguru import logger

from datenwissenschaften.curriculum import atomic_write
from datenwissenschaften.environment.demonstration import TAS_DIR, TAS_MOVIE, TAS_PROGRESS
from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.levels import level_map
from datenwissenschaften.environment.wrapper import phases
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, load_config
from datenwissenschaften.tas.movie import Segment, verified_segments
from datenwissenschaften.tas.report import SECTION, report
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import UNBOUNDED, Crossing, fastest_distinct
from datenwissenschaften.tas.workers import explore, polish
from datenwissenschaften.training.session import MAIN_ENVIRONMENT
from datenwissenschaften.training.video_render import render_video
from datenwissenschaften.ui.telemetry import publish_metadata
from datenwissenschaften.workflow.wrapper import wrapper_class

PROGRESS_FILE = TAS_PROGRESS
LOG_FILE = "search.log"
SEARCH_ROUNDS = 3
IMPROVE_ROUNDS = 1

app = typer.Typer()


def load_progress(path: Path) -> list[Segment]:
    return msgspec.msgpack.decode(path.read_bytes(), type=list[Segment]) if path.is_file() else []


def save_progress(path: Path, segments: list[Segment]) -> None:
    atomic_write(path, msgspec.msgpack.encode(segments))


def finished(segments: list[Segment], until: str) -> bool:
    return bool(segments) and (segments[-1].won or segments[-1].state == until)


def frames(segments: list[Segment]) -> int:
    return sum(len(segment.pressed) for segment in segments)


def segment_of(crossing: Crossing) -> Segment:
    return Segment(crossing.end.state, crossing.pressed.tolist(), crossing.end.emulator_state, crossing.won)


def tas_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / TAS_DIR


@dataclass(slots=True)
class Tas:
    wrapper: str
    config: Path
    settings: RetroSpeedlabConfig
    runner: SegmentRunner
    pool: Executor

    @property
    def output(self) -> Path:
        return tas_dir(self.settings)

    def start(self, segments: list[Segment], index: int) -> Start:
        if index == 0:
            return self.runner.power_on()
        return Start(segments[index - 1].state, segments[index - 1].emulator_state)

    def candidates(self, start: Start, bound: int, rounds: int) -> list[Crossing]:
        workers = self.settings.speedrun.workers
        for round_index in range(rounds):
            seconds = self.settings.speedrun.segment_seconds * 2**round_index
            seeds = range(round_index * workers, (round_index + 1) * workers)
            searches = [self.pool.submit(explore, *self._job(start, bound, seed, seconds)) for seed in seeds]
            if found := fastest_distinct([crossing for search in searches for crossing in search.result()]):
                return found
        return []

    def best(self, start: Start, bound: int, rounds: int) -> Crossing | None:
        found = self.candidates(start, bound, rounds)
        if not found:
            return None
        seconds = self.settings.speedrun.segment_seconds
        ahead = [
            self.pool.submit(explore, *self._job(crossing.end, UNBOUNDED, seed, seconds))
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
        return self.report(segments, searching)

    def report(self, segments: list[Segment], searching: str) -> list[Segment]:
        levels = level_map(self.runner.wrapper.levels, self.runner.wrapper.state_classes)
        frame_rate = self.runner.emulator.em.get_screen_rate()
        records = self.settings.speedrun.records
        publish_metadata(
            SECTION, report(segments, levels, self.runner.order[0], frame_rate, records, searching), replace=True
        )
        return segments

    def verified(self, segments: list[Segment], shown: int) -> list[Segment]:
        verified = verified_segments(self.runner, segments, self.output / TAS_MOVIE, shown)
        if verified == len(segments):
            return segments
        logger.warning(f"Only {verified} of {len(segments)} segments replay from power-on; searching again from there")
        return segments[:verified]

    def extend(self, segments: list[Segment], until: str) -> list[Segment]:
        while not finished(segments := self.verified(segments, len(segments) - 1), until):
            start = self.start(segments, len(segments))
            self.report(segments, start.state)
            crossing = self.best(start, UNBOUNDED, SEARCH_ROUNDS)
            if crossing is None:
                raise RuntimeError(f"The search found no way out of {start.state} in {SEARCH_ROUNDS} rounds")
            segments = self.keep([*segments, segment_of(crossing)], crossing.end.state)
            logger.info(f"{start.state} -> {crossing.end.state} in {crossing.frames} frames; run at {frames(segments)}")
        return self.report(segments, segments[-1].state)

    def improve(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            raise RuntimeError(f"{self.output / PROGRESS_FILE} holds no movie to improve; search one first")
        until, index = segments[-1].state, 0
        while index < len(segments):
            start = self.start(segments, index)
            crossing = self.best(start, len(segments[index].pressed), IMPROVE_ROUNDS)
            if crossing is not None and crossing.frames < len(segments[index].pressed):
                saved = len(segments[index].pressed) - crossing.frames
                logger.info(f"Segment {index} into {crossing.end.state} is {saved} frames faster; the rest is rebuilt")
                segments = self.extend(self.keep([*segments[:index], segment_of(crossing)], start.state), until)
            index += 1
        return segments

    def _job(self, start: Start, bound: int, seed: int, seconds: float) -> tuple[str, Path, Start, int, int, float]:
        return self.wrapper, self.config, start, bound, seed, seconds


def publish(settings: RetroSpeedlabConfig, segments: list[Segment]) -> None:
    movie = tas_dir(settings) / TAS_MOVIE
    movie.with_suffix(".mp4").unlink(missing_ok=True)
    video = render_video(settings.paths, movie)
    logger.info(f"Verified movie {movie} reaches {segments[-1].state} in {frames(segments)} frames; video {video}")


def worker_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def tas_run(wrapper: str, config: Path, pool: Executor) -> Tas:
    settings = load_config(config)
    setup_logging(settings.log_level)
    tas_dir(settings).mkdir(parents=True, exist_ok=True)
    logger.add(tas_dir(settings) / LOG_FILE)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_ENVIRONMENT, False))
    return Tas(wrapper, config, settings, runner, pool)


def speedrun_forever(wrapper: str, config: Path) -> None:
    with worker_pool(load_config(config).speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        segments = tas.extend(load_progress(tas.output / PROGRESS_FILE), phases(wrapper_class(wrapper))[-1])
        publish(tas.settings, segments)
        while True:
            improved = tas.improve(segments)
            if frames(improved) < frames(segments):
                publish(tas.settings, improved)
            segments = improved


@app.command()
def search(wrapper: str, config: Path, until: str) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        segments = tas.extend(load_progress(tas.output / PROGRESS_FILE), until)
    publish(settings, segments)


@app.command()
def improve(wrapper: str, config: Path) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        segments = tas.improve(tas.verified(load_progress(tas.output / PROGRESS_FILE), 0))
    publish(settings, segments)


@app.command()
def verify(wrapper: str, config: Path, progress: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    segments = load_progress(progress)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_ENVIRONMENT, False))
    verified = verified_segments(runner, segments, tas_dir(settings) / f"verify-{TAS_MOVIE}", len(segments))
    if verified < len(segments):
        raise RuntimeError(f"Only {verified} of {len(segments)} segments of {progress} replay with {wrapper}")
    logger.info(f"All {len(segments)} segments of {progress} replay with {wrapper}")


if __name__ == "__main__":
    app()
