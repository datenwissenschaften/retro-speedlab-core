import random
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger

from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.tas.search import KEEP_ACTION, Cell, choose

SAMPLES_PER_LEVEL = 6000
SAMPLE_EVERY_FRAMES = 3
WALK_DECISIONS = 40
START_EVERY_FRAMES = 30
SIDE_CROP = 8
SCALE = 2
NOT_PLAYING = -1

Archive = dict[CellKey, Cell]


@dataclass(slots=True)
class Samples:
    screens: list[np.ndarray]
    labels: list[tuple[int, int, int, int]]
    frames: int


def small_screen(screen: np.ndarray, side_crop: int) -> np.ndarray:
    cropped = screen[:, side_crop : screen.shape[1] - side_crop] if side_crop else screen
    return cv2.resize(cropped, (cropped.shape[1] // SCALE, cropped.shape[0] // SCALE), interpolation=cv2.INTER_AREA)


def current_level(runner: SegmentRunner) -> int:
    address = runner.wrapper.level_address
    return 0 if address is None else int(runner.emulator.get_ram()[address])


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


def hold_level(runner: SegmentRunner, level: int) -> None:
    if runner.wrapper.level_address is not None:
        runner.emulator.data.memory.assign(runner.wrapper.level_address, "|u1", level)


def press_held(runner: SegmentRunner, frames: np.ndarray, level: int, samples: Samples) -> bool:
    for buttons in frames:
        hold_level(runner, level)
        if not press(runner, buttons, samples):
            return False
    return True


def boot_frames(runner: SegmentRunner) -> np.ndarray:
    frames = np.zeros((START_EVERY_FRAMES, len(runner.emulator.buttons)), dtype=np.int8)
    frames[0, runner.emulator.buttons.index("START")] = 1
    return frames


def remember(runner: SegmentRunner, archive: Archive, frames: int) -> None:
    key = runner.key(frames)
    if runner.ram.position() is not None and key not in archive:
        start = Start(runner.wrapper.state_machine.state_name, runner.snapshot())
        archive[key] = Cell(start, None, frames, 0, None, float(frames))


def replay_route(runner: SegmentRunner, route: list[Segment], samples: Samples) -> dict[int, Archive]:
    seen: dict[int, Archive] = defaultdict(dict)
    runner.power_on()
    frames = [np.asarray(segment.pressed, dtype=np.int8) for segment in route]
    for played, buttons in enumerate(np.concatenate(frames) if frames else []):
        press(runner, buttons, samples)
        remember(runner, seen[current_level(runner)], played)
    return seen


def walk(runner: SegmentRunner, level: int, archive: Archive, samples: Samples, rng: random.Random) -> None:
    cell = choose(list(archive.values()), rng) if archive else None
    frames = 0
    if cell is None:
        runner.power_on()
    else:
        runner.restore(cell.start)
        frames = cell.frames
    choice = rng.random()
    for _ in range(WALK_DECISIONS):
        if runner.ram.position() is None:
            pressed = boot_frames(runner)
        else:
            actions = runner.wrapper.current_action_table()
            choice = choice if rng.random() < KEEP_ACTION else rng.random()
            pressed = actions[int(choice * len(actions))]
        if not press_held(runner, pressed, level, samples):
            return
        frames += len(pressed)
        remember(runner, archive, frames)


def explore_level(runner: SegmentRunner, level: int, archive: Archive, samples: Samples, rng: random.Random) -> None:
    goal = len(samples.labels) + SAMPLES_PER_LEVEL
    while len(samples.labels) < goal:
        walk(runner, level, archive, samples, rng)
    logger.info(f"Level {level}: {len(archive)} places seen")


def collect_samples(runner: SegmentRunner, route: list[Segment], seed: int) -> Samples:
    samples, rng = Samples([], [], 0), random.Random(seed)
    seen = replay_route(runner, route, samples)
    for level in range(runner.wrapper.level_count):
        explore_level(runner, level, seen[level], samples, rng)
    return samples
