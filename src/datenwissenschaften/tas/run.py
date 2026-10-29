import multiprocessing
import sys
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import msgspec
import typer
from loguru import logger

from datenwissenschaften.curriculum import atomic_write
from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.movie import Segment, verified_segments
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import Crossing
from datenwissenschaften.tas.workers import explore
from datenwissenschaften.training.session import MAIN_ENVIRONMENT
from datenwissenschaften.training.video_render import render_video
from datenwissenschaften.workflow.wrapper import wrapper_class

PROGRESS_FILE = "progress.msgpack"
MOVIE_FILE = "tas.bk2"
SEARCH_ROUNDS = 3
IMPROVE_ROUNDS = 1
UNBOUNDED = sys.maxsize

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


@dataclass(slots=True)
class Tas:
    wrapper: str
    config: Path
    output: Path
    workers: int
    seconds: float
    runner: SegmentRunner
    pool: Executor

    def start(self, segments: list[Segment], index: int) -> Start:
        if index == 0:
            return self.runner.power_on()
        return Start(segments[index - 1].state, segments[index - 1].emulator_state)

    def fastest(self, start: Start, bound: int, rounds: int) -> Crossing | None:
        for round_index in range(rounds):
            seeds = range(round_index * self.workers, (round_index + 1) * self.workers)
            searches = [
                self.pool.submit(explore, self.wrapper, self.config, start, bound, seed, self.seconds) for seed in seeds
            ]
            found = [crossing for search in searches if (crossing := search.result()) is not None]
            if found:
                return min(found, key=lambda crossing: crossing.frames)
        return None

    def keep(self, segments: list[Segment]) -> list[Segment]:
        save_progress(self.output / PROGRESS_FILE, segments)
        return segments

    def verified(self, segments: list[Segment]) -> list[Segment]:
        verified = verified_segments(self.runner, segments, self.output / MOVIE_FILE)
        if verified == len(segments):
            return segments
        logger.warning(f"Only {verified} of {len(segments)} segments replay from power-on; searching again from there")
        return self.keep(segments[:verified])

    def extend(self, segments: list[Segment], until: str) -> list[Segment]:
        while not finished(segments := self.verified(segments), until):
            start = self.start(segments, len(segments))
            crossing = self.fastest(start, UNBOUNDED, SEARCH_ROUNDS)
            if crossing is None:
                raise RuntimeError(f"The search found no way out of {start.state} in {SEARCH_ROUNDS} rounds")
            segments = self.keep([*segments, segment_of(crossing)])
            logger.info(f"{start.state} -> {crossing.end.state} in {crossing.frames} frames; run at {frames(segments)}")
        return segments

    def improve(self, segments: list[Segment]) -> list[Segment]:
        if not segments:
            raise RuntimeError(f"{self.output / PROGRESS_FILE} holds no movie to improve; search one first")
        until, index = segments[-1].state, 0
        while index < len(segments):
            crossing = self.fastest(self.start(segments, index), len(segments[index].pressed), IMPROVE_ROUNDS)
            if crossing is not None:
                saved = len(segments[index].pressed) - crossing.frames
                logger.info(f"Segment {index} into {crossing.end.state} is {saved} frames faster; the rest is rebuilt")
                segments = self.extend(self.keep([*segments[:index], segment_of(crossing)]), until)
            index += 1
        return segments


def publish(config: Path, output: Path, segments: list[Segment]) -> None:
    movie = output / MOVIE_FILE
    movie.with_suffix(".mp4").unlink(missing_ok=True)
    video = render_video(load_config(config).paths, movie)
    logger.info(f"Verified movie {movie} reaches {segments[-1].state} in {frames(segments)} frames; video {video}")


def worker_pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def tas_run(wrapper: str, config: Path, output: Path, workers: int, seconds: float, pool: Executor) -> Tas:
    settings = load_config(config)
    setup_logging(settings.log_level)
    output.mkdir(parents=True, exist_ok=True)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings, MAIN_ENVIRONMENT, False))
    return Tas(wrapper, config, output, workers, seconds, runner, pool)


@app.command()
def search(wrapper: str, config: Path, output: Path, until: str, workers: int, segment_seconds: float) -> None:
    with worker_pool(workers) as pool:
        tas = tas_run(wrapper, config, output, workers, segment_seconds, pool)
        segments = tas.extend(load_progress(output / PROGRESS_FILE), until)
    publish(config, output, segments)


@app.command()
def improve(wrapper: str, config: Path, output: Path, workers: int, segment_seconds: float) -> None:
    with worker_pool(workers) as pool:
        tas = tas_run(wrapper, config, output, workers, segment_seconds, pool)
        segments = tas.improve(tas.verified(load_progress(output / PROGRESS_FILE)))
    publish(config, output, segments)


if __name__ == "__main__":
    app()
