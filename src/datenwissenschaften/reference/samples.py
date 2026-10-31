from collections import Counter, defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from datenwissenschaften.route.route_map import Cell, cell_of
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import SegmentRunner

SAMPLES_PER_LEVEL = 6000
SAMPLE_EVERY_FRAMES = 3
WALK_DECISIONS = 40
START_EVERY_FRAMES = 30
SIDE_CROP = 8
SCALE = 2
NO_POSITION = (-1, 0, 0, 0)

Archive = dict[Cell, bytes]


@dataclass(slots=True)
class Samples:
    screens: list[np.ndarray]
    labels: list[tuple[int, int, int, int]]


def small_screen(screen: np.ndarray, side_crop: int) -> np.ndarray:
    cropped = screen[:, side_crop : screen.shape[1] - side_crop] if side_crop else screen
    return cv2.resize(cropped, (cropped.shape[1] // SCALE, cropped.shape[0] // SCALE), interpolation=cv2.INTER_AREA)


def record(runner: SegmentRunner, samples: Samples) -> None:
    position = runner.ram.position()
    samples.screens.append(small_screen(runner.emulator.em.get_screen(), 0))
    samples.labels.append(NO_POSITION if position is None else (position.area, position.x, position.y, position.height))


def current_level(runner: SegmentRunner) -> int:
    address = runner.wrapper.level_address
    return 0 if address is None else int(runner.emulator.get_ram()[address])


def hold_level(runner: SegmentRunner, level: int) -> None:
    if runner.wrapper.level_address is not None:
        runner.emulator.data.memory.assign(runner.wrapper.level_address, "|u1", level)


def press_frames(runner: SegmentRunner, frames: np.ndarray, level: int, samples: Samples, played: int) -> int:
    for buttons in frames:
        hold_level(runner, level)
        runner.press(buttons)
        played += 1
        if played % SAMPLE_EVERY_FRAMES == 0:
            record(runner, samples)
    return played


def boot_frames(runner: SegmentRunner) -> np.ndarray:
    frames = np.zeros((START_EVERY_FRAMES, len(runner.emulator.buttons)), dtype=np.int8)
    frames[0, runner.emulator.buttons.index("START")] = 1
    return frames


def replay_route(runner: SegmentRunner, route: list[Segment], samples: Samples) -> dict[int, Archive]:
    seen: dict[int, Archive] = defaultdict(dict)
    runner.power_on()
    frames = [np.asarray(segment.pressed, dtype=np.int8) for segment in route]
    for played, buttons in enumerate(np.concatenate(frames) if frames else []):
        runner.press(buttons)
        position = runner.ram.position()
        if position is not None:
            seen[current_level(runner)].setdefault(cell_of(position), runner.snapshot())
        if played % SAMPLE_EVERY_FRAMES == 0:
            record(runner, samples)
    return seen


def restore_least_visited(runner: SegmentRunner, archive: Archive, visits: Counter, rng: np.random.Generator) -> None:
    cells = list(archive)
    weights = np.asarray([1 / np.sqrt(1 + visits[cell]) for cell in cells])
    cell = cells[rng.choice(len(cells), p=weights / weights.sum())]
    visits[cell] += 1
    runner.emulator.em.set_state(archive[cell])
    runner.emulator.data.update_ram()
    runner.ram = runner.wrapper.read_ram()


def explore_level(
    runner: SegmentRunner, level: int, archive: Archive, samples: Samples, rng: np.random.Generator
) -> None:
    visits: Counter = Counter()
    runner.power_on()
    played, goal = 0, len(samples.labels) + SAMPLES_PER_LEVEL
    while len(samples.labels) < goal:
        if archive:
            restore_least_visited(runner, archive, visits, rng)
        for _ in range(WALK_DECISIONS):
            position = runner.ram.position()
            if position is not None:
                archive.setdefault(cell_of(position), runner.snapshot())
            actions = runner.wrapper.current_action_table()
            frames = boot_frames(runner) if position is None else actions[rng.integers(len(actions))]
            played = press_frames(runner, frames, level, samples, played)
    logger.info(f"Level {level}: {len(archive)} places seen in {played} frames")


def collect_samples(runner: SegmentRunner, route: list[Segment], seed: int) -> Samples:
    samples, rng = Samples([], []), np.random.default_rng(seed)
    seen = replay_route(runner, route, samples)
    for level in range(runner.wrapper.level_count):
        explore_level(runner, level, seen[level], samples, rng)
    return samples
