import multiprocessing
import os
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.experiments import start_experiment
from datenwissenschaften.knowledge import knowledge_path, load_knowledge
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig, RetroSpeedlabPaths, load_config
from datenwissenschaften.speedrun.feed import SpeedrunFeed, publish_every
from datenwissenschaften.speedrun.footage import compare_cut, movie_cut, render_compare, render_movie
from datenwissenschaften.speedrun.hints import Hint, hinted, hints_path, load_hints, pressed
from datenwissenschaften.speedrun.journal import JOURNAL_FILE, JournalStore
from datenwissenschaften.speedrun.live import LIVE_DIR
from datenwissenschaften.speedrun.movie import Segment, frames, verified_segments
from datenwissenschaften.speedrun.paths import PROGRESS_FILE, TAS_MOVIE, tas_dir
from datenwissenschaften.speedrun.policy_start import (
    POLICY_EXITS_FILE,
    POLICY_START_FILE,
    load_policy_exits,
    save_policy_start,
)
from datenwissenschaften.speedrun.report import course_of
from datenwissenschaften.speedrun.routes import FULL_RUN, ROUTES_FILE, Leg, Routes, finished, load_progress, parts_of
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.screen import Screen
from datenwissenschaften.speedrun.search import ROUTE_CHECKPOINT, UNBOUNDED, Checkpoint, Crossing
from datenwissenschaften.speedrun.searcher import Searcher
from datenwissenschaften.speedrun.shown import SHOWN_FILE, worth_showing
from datenwissenschaften.speedrun.stuck import (
    BARRED_FILE,
    FAILURES_FILE,
    STUCK_FILE,
    bar,
    cleared,
    failed,
    load_barred,
    load_failures,
    load_stuck,
    mark_stuck,
    save_stuck,
    unbar,
)
from datenwissenschaften.speedrun.turns import TURN_FILE, Turn, save_turn, turn_of
from datenwissenschaften.sprites.catalog import SpriteCatalog, sprites_dir
from datenwissenschaften.ui.relay import relay_key
from datenwissenschaften.ui.video_upload import VideoUpload
from datenwissenschaften.workflow.wrapper import wrapper_class

LOG_FILE = "search.log"
SEARCH_ROUNDS = 2
BACKTRACK_SEARCHES = 3
PRACTICE_BACKTRACK_TURNS = 3
IMPROVE_ROUNDS = 1

MAIN_WORKER = 0
SEARCH_JOB = "search"
MATH_THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
TRIAL_MOVIE = f"trial-{TAS_MOVIE}"

Footage = Callable[[RetroSpeedlabPaths, Path, tuple[bytes, ...]], Path]

app = typer.Typer()


def never_rest() -> bool:
    return False


def segment_of(crossing: Crossing) -> Segment:
    return Segment(crossing.end.state, crossing.pressed.tolist(), crossing.end.emulator_state, crossing.won)


def checkpoint_segment(checkpoint: Checkpoint) -> Segment:
    return Segment(checkpoint.end.state, checkpoint.pressed.tolist(), checkpoint.end.emulator_state, False)


def turn_searches(patience: int | None) -> int:
    return BACKTRACK_SEARCHES if patience is None else patience


def backtrack_searches(patience: int | None) -> int:
    return BACKTRACK_SEARCHES if patience is None else patience * PRACTICE_BACKTRACK_TURNS


def checkpoints_in_a_row(leg: Leg, segments: list[Segment]) -> int:
    states = [leg.entry.state, *(segment.state for segment in segments)]
    count = 0
    while count < len(segments) and states[-1 - count] == states[-2 - count]:
        count += 1
    return count


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
    hints: list[Hint]
    hinted: set[tuple[str, str]]

    @property
    def output(self) -> Path:
        return tas_dir(self.settings)

    def start(self, leg: Leg, segments: list[Segment], index: int) -> Start:
        if index == 0:
            return leg.entry
        return Start(segments[index - 1].state, segments[index - 1].emulator_state)

    @property
    def stuck_file(self) -> Path:
        return self.output / STUCK_FILE

    @property
    def turn_file(self) -> Path:
        return self.output / TURN_FILE

    @property
    def barred_file(self) -> Path:
        return self.output / BARRED_FILE

    @property
    def barred(self) -> dict[str, str]:
        return load_barred(self.barred_file)

    def hint(self, leg: Leg, segments: list[Segment], start: Start) -> tuple[list[Segment], Crossing] | None:
        if (leg.name, start.state) in self.hinted:
            return None
        starts = [self.start(leg, segments, index) for index in range(len(segments) + 1)]
        lab = [pressed(self.runner, hint) for hint in self.hints if hint.state == start.state]
        policy = load_policy_exits(self.output / POLICY_EXITS_FILE)
        tried = [*lab, *([np.asarray(policy[start.state], dtype=np.int8)] if start.state in policy else [])]
        found = hinted(self.runner, tried, starts)
        if found is None:
            return None
        self.hinted.add((leg.name, start.state))
        index, crossing = found
        logger.info(f"A hint leads out of {start.state} into {crossing.end.state}")
        return segments[:index], crossing

    def leave(self, state: str) -> None:
        stuck = load_stuck(self.stuck_file)
        if stuck is not None and stuck.state != state:
            self.stuck_file.unlink()

    def resume_stuck(self, state: str) -> None:
        failures = load_failures(self.output / FAILURES_FILE)
        if state in failures:
            save_stuck(self.stuck_file, failures[state])
        else:
            self.stuck_file.unlink(missing_ok=True)

    def keep(self, leg: Leg, segments: list[Segment], searching: str) -> list[Segment]:
        self.routes.save(leg, segments)
        self.feed.show(self.routes.view(leg.name, segments), searching)
        return segments

    def verified(self, leg: Leg, segments: list[Segment]) -> list[Segment]:
        movie = self.output / TAS_MOVIE
        replay = verified_segments(self.runner, leg.entry, segments, movie, self.feed.catalog)
        replayed = [
            replace(segment, emulator_state=end)
            for segment, end in zip(segments[: replay.verified], replay.ends, strict=True)
        ]
        if replay.verified < len(segments):
            logger.warning(f"Only {replay.verified} of {len(segments)} segments of {leg.name} replay; searching again")
        elif frames(replayed) and worth_showing(self.output / SHOWN_FILE, self.runner.order, leg.entry.state, replayed):
            self.movie.queue(movie_cut(self.feed.course, leg, replayed, movie, replay.memory))
        return replayed

    def extend(
        self, leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]
    ) -> list[Segment]:
        turn = Turn(leg.name, 0) if patience is None else turn_of(self.turn_file, leg.name, patience)
        while not finished(segments := self.verified(leg, segments), leg.exit):
            if patience is not None and turn.over(patience):
                logger.info(f"{leg.name} rests after {turn.searches} searches; the next level takes its turn")
                return segments
            if rest():
                return segments
            turn = turn.searched()
            start = self.start(leg, segments, len(segments))
            if patience is not None:
                self.resume_stuck(start.state)
            save_policy_start(self.output / POLICY_START_FILE, start)
            self.feed.show(self.routes.view(leg.name, segments), start.state)
            hint = self.hint(leg, segments, start)
            if hint is None:
                found = self.searcher.best(start, UNBOUNDED, SEARCH_ROUNDS)
            else:
                segments, found = hint
            if patience is not None:
                save_turn(self.turn_file, turn)
            if isinstance(found, Checkpoint):
                segments = self.checkpoint(leg, segments, start, found, patience)
                continue
            if found is None:
                segments = self.stuck(leg, segments, start, patience)
                continue
            cleared(self.output / FAILURES_FILE, start.state)
            unbar(self.barred_file, leg.name, start.state)
            turn = turn.found()
            if patience is not None:
                save_turn(self.turn_file, turn)
            self.leave(found.end.state)
            self.feed.event("found", found.end.state, found.frames)
            segments = self.keep(leg, [*segments, segment_of(found)], found.end.state)
            logger.info(
                f"{start.state} -> {found.end.state} in {found.frames} frames; {leg.name} at {frames(segments)}"
            )
            segments = self.shorten(leg, segments, self.turn % len(segments))
            self.turn += 1
        self.feed.show(self.routes.view(leg.name, segments), segments[-1].state)
        return segments

    def searched(self, state: str, patience: int | None) -> int:
        if patience is None:
            return mark_stuck(self.stuck_file, state).searches
        return save_stuck(self.stuck_file, failed(self.output / FAILURES_FILE, state)).searches

    def closer(self, start: Start, found: Checkpoint) -> bool:
        if found.remaining is None:
            return False
        self.runner.restore(start)
        before = self.runner.remaining()
        return before is None or found.remaining <= before - ROUTE_CHECKPOINT

    def checkpoint(
        self, leg: Leg, segments: list[Segment], start: Start, found: Checkpoint, patience: int | None
    ) -> list[Segment]:
        searches = self.searched(start.state, patience)
        held = start.state == self.barred.get(leg.name) or searches >= turn_searches(patience)
        if held and not self.closer(start, found):
            return self.wait_or_go_back(leg, segments, start, searches, patience)
        self.feed.event("checkpoint", start.state, found.frames)
        if found.remaining is not None:
            self.feed.journal.closer(start.state, found.remaining)
        on = f"{found.remaining} TAS frames before the exit" if found.guided else f"{found.gain:.1f} reward on"
        logger.info(f"Checkpoint in {start.state} after {found.frames} frames, {on}")
        return self.keep(leg, [*segments, checkpoint_segment(found)], start.state)

    def stuck(self, leg: Leg, segments: list[Segment], start: Start, patience: int | None) -> list[Segment]:
        return self.wait_or_go_back(leg, segments, start, self.searched(start.state, patience), patience)

    def wait_or_go_back(
        self, leg: Leg, segments: list[Segment], start: Start, searches: int, patience: int | None
    ) -> list[Segment]:
        self.feed.event("stuck", start.state, None)
        chain = checkpoints_in_a_row(leg, segments)
        if chain and searches >= turn_searches(patience):
            logger.warning(f"No way out of {start.state} after {searches} searches; its {chain} checkpoints go")
        elif searches % backtrack_searches(patience):
            logger.warning(f"No way out of {start.state} after {searches} searches; searching again")
            return segments
        else:
            logger.warning(f"No way out of {start.state} after {searches} searches; searching the way into it again")
        if segments and segments[-1].state == start.state:
            bar(self.barred_file, leg.name, start.state)
        kept = segments[: len(segments) - max(1, chain)]
        return self.keep(leg, kept, self.start(leg, kept, len(kept)).state)

    def improve(self, leg: Leg, segments: list[Segment]) -> list[Segment]:
        for index in range(len(segments)):
            segments = self.shorten(leg, segments, index)
        return segments

    def shorten(self, leg: Leg, segments: list[Segment], index: int) -> list[Segment]:
        start = self.start(leg, segments, index)
        crossing = self.searcher.best(start, len(segments[index].pressed), IMPROVE_ROUNDS)
        if not isinstance(crossing, Crossing) or crossing.frames >= len(segments[index].pressed):
            return segments
        saved = len(segments[index].pressed) - crossing.frames
        shorter = [*segments[:index], segment_of(crossing), *segments[index + 1 :]]
        if verified_segments(self.runner, leg.entry, shorter, self.output / TRIAL_MOVIE, None).verified < len(shorter):
            logger.info(f"Segment {index} of {leg.name} is {saved} frames faster but the rest breaks; it stays")
            return segments
        logger.info(f"Segment {index} of {leg.name} into {crossing.end.state} is {saved} frames faster")
        self.feed.event("improved", crossing.end.state, crossing.frames)
        self.feed.stats.trimmed(saved)
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
    os.environ.update(dict.fromkeys(MATH_THREADS, "1"))
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
    experiment = start_experiment(settings, SEARCH_JOB)
    catalog = SpriteCatalog(sprites_dir(settings), load_knowledge(knowledge_path(settings.paths.game_dir)))
    feed = SpeedrunFeed(
        course, journal, output / LIVE_DIR, output / STUCK_FILE, settings.paths.lab_run_marker, experiment, catalog
    )
    lab_running = settings.paths.lab_run_marker.exists()
    stopped = threading.Event()
    publishing = threading.Thread(target=publish_every, args=(feed, stopped, lab_running), name="speedrun-feed")
    publishing.daemon = True
    publishing.start()
    movie, compare = screen(settings, feed, "movie", render_movie), screen(settings, feed, "compare", render_compare)
    searcher = Searcher(wrapper, config, settings, pool, feed)
    routes = Routes(output / ROUTES_FILE, output / PROGRESS_FILE, parts_of(course))
    if relabelled := routes.relabel(runner):
        logger.warning(
            f"Pieces of {', '.join(relabelled)} reach other states with this game package; they keep their place"
        )
    return Tas(settings, runner, searcher, routes, feed, movie, compare, 0, load_hints(hints_path(config)), set())


@app.command()
def search(wrapper: str, config: Path, until: str) -> None:
    settings = load_config(config)
    with worker_pool(settings.speedrun.workers) as pool:
        tas = tas_run(wrapper, config, pool)
        leg = Leg(FULL_RUN, tas.runner.power_on(), until)
        publish(tas, tas.extend(leg, load_progress(tas.routes.progress), None, never_rest))


def kept_on_verify(leg: Leg, segments: list[Segment]) -> int:
    return len(segments) if finished(segments, leg.exit) else len(segments) - 1


@app.command()
def verify(wrapper: str, config: Path, routes_file: Path) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    hints = load_hints(hints_path(config))
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
        verified = verified_segments(
            runner, leg.entry, segments, tas_dir(settings) / f"verify-{TAS_MOVIE}", None
        ).verified
        if verified < kept_on_verify(leg, segments):
            raise RuntimeError(f"Only {verified} of {len(segments)} segments of {leg.name} replay with {wrapper}")
        if verified < len(segments):
            logger.warning(f"The newest segment of {leg.name} into {segments[-1].state} no longer replays")
    logger.info(f"Every part of {routes_file} replays with {wrapper}; {len(hints)} hints read")


if __name__ == "__main__":
    app()
