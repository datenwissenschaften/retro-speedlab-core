import random
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.tas.search import KEEP_ACTION, Cell, choose

SAMPLES_PER_LEVEL = 5000
SAMPLE_EVERY_FRAMES = 3
WALK_DECISIONS = 40
ROLLOUT_DECISIONS = 6
START_EVERY_FRAMES = 30
SIDE_CROP = 8
SCALE = 2
NOT_PLAYING = -1

Archive = dict[CellKey, Cell]
Trail = tuple[Start, list[Segment]]


@dataclass(slots=True)
class Samples:
    screens: list[np.ndarray]
    labels: list[tuple[int, int, int, int]]
    frames: int


def small_screen(screen: np.ndarray, side_crop: int) -> np.ndarray:
    cropped = screen[:, side_crop : screen.shape[1] - side_crop] if side_crop else screen
    return cv2.resize(cropped, (cropped.shape[1] // SCALE, cropped.shape[0] // SCALE), interpolation=cv2.INTER_AREA)


def current_level(runner: SegmentRunner) -> int:
    level = runner.ram.level()
    return 0 if level is None else level


def press(runner: SegmentRunner, buttons: np.ndarray, samples: Samples) -> bool:
    alive, _, _ = runner.press(buttons)
    samples.frames += 1
    if samples.frames % SAMPLE_EVERY_FRAMES == 0:
        position, level = runner.ram.position(), current_level(runner)
        samples.screens.append(small_screen(runner.emulator.em.get_screen(), 0))
        samples.labels.append(
            (NOT_PLAYING, 0, 0, level) if position is None else (position.area, position.x, position.y, level)
        )
    return alive


def press_all(runner: SegmentRunner, frames: np.ndarray, samples: Samples) -> bool:
    return all(press(runner, buttons, samples) for buttons in frames)


def boot_frames(runner: SegmentRunner) -> np.ndarray:
    frames = np.zeros((START_EVERY_FRAMES, len(runner.emulator.buttons)), dtype=np.int8)
    frames[0, runner.emulator.buttons.index("START")] = 1
    return frames


def remember(runner: SegmentRunner, archive: Archive, frames: int) -> None:
    key = runner.key(frames)
    if runner.ram.position() is not None and key not in archive:
        start = Start(runner.wrapper.state_machine.state_name, runner.snapshot())
        archive[key] = Cell(start, None, frames, 0, None, float(frames), 0)


def replay(runner: SegmentRunner, trail: Trail, samples: Samples, seen: dict[int, Archive]) -> None:
    start, route = trail
    runner.restore(start)
    for played, buttons in enumerate(buttons for segment in route for buttons in segment.pressed):
        press(runner, np.asarray(buttons, dtype=np.int8), samples)
        remember(runner, seen[current_level(runner)], played)


def roll(
    runner: SegmentRunner, archive: Archive, samples: Samples, rng: random.Random, frames: int, decisions: int
) -> None:
    choice = rng.random()
    for _ in range(decisions):
        if runner.ram.position() is None:
            pressed = boot_frames(runner)
        else:
            actions = runner.wrapper.current_action_table()
            choice = choice if rng.random() < KEEP_ACTION else rng.random()
            pressed = actions[int(choice * len(actions))]
        if not press_all(runner, pressed, samples):
            return
        frames += len(pressed)
        remember(runner, archive, frames)


def walk(runner: SegmentRunner, archive: Archive, samples: Samples, rng: random.Random) -> None:
    cell = choose(list(archive.values()), rng) if archive else None
    if cell is None:
        runner.power_on()
    else:
        runner.restore(cell.start)
    roll(runner, archive, samples, rng, 0 if cell is None else cell.frames, WALK_DECISIONS)


def roll_out(
    runner: SegmentRunner, explored: list[Start], samples: Samples, seen: dict[int, Archive], rng: random.Random
) -> None:
    before = len(samples.labels)
    for start in explored:
        runner.restore(start)
        roll(runner, seen[current_level(runner)], samples, rng, 0, ROLLOUT_DECISIONS)
    logger.info(f"{len(explored)} places the search explored add {len(samples.labels) - before} training screens")


def explore_level(runner: SegmentRunner, level: int, archive: Archive, samples: Samples, rng: random.Random) -> None:
    goal = len(samples.labels) + SAMPLES_PER_LEVEL
    while len(samples.labels) < goal:
        walk(runner, archive, samples, rng)
    logger.info(f"Level {level}: {len(archive)} places seen")


def entered(runner: SegmentRunner, start: Start, archive: Archive) -> Archive:
    runner.restore(start)
    remember(runner, archive, 0)
    return archive


def collect_samples(
    runner: SegmentRunner, trails: list[Trail], starts: list[Start], explored: list[Start], seed: int
) -> Samples:
    samples, rng, seen = Samples([], [], 0), random.Random(seed), defaultdict[int, Archive](dict)
    for trail in trails:
        replay(runner, trail, samples, seen)
    roll_out(runner, explored, samples, seen, rng)
    explore_level(runner, 0, seen[0], samples, rng)
    for level, start in enumerate(starts, 1):
        explore_level(runner, level, entered(runner, start, seen[level]), samples, rng)
    return samples
