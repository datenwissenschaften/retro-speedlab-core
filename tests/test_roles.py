import subprocess
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from fakes import saving_environment

from datenwissenschaften.speedrun import vision
from datenwissenschaften.speedrun.beam import beam_search
from datenwissenschaften.speedrun.improver import SMALLEST_GAIN_FRAMES, Improver, Repair, shortenable
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import Leg
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing, Searched

BOSS_FRAMES = 3
SECONDS = 0.5
LONG = SMALLEST_GAIN_FRAMES * 3


class RecordingPool:
    def __init__(self) -> None:
        self.jobs: list[tuple[object, ...]] = []
        self.futures: list[Future] = []

    def submit(self, function, *arguments) -> Future:
        self.jobs.append(arguments)
        self.futures.append(Future())
        return self.futures[-1]


def piece(state: str, frames: int, end: int) -> Segment:
    return Segment(state, [[0, 0]] * frames, bytes([end]), False)


def leg() -> Leg:
    return Leg("Level 1", Start("Survive", bytes([0])), "Exit")


def crossing(state: str, frames: int) -> Crossing:
    return Crossing(Start(state, bytes([9])), np.zeros((frames, 2), np.int8), False, ("cell",))


def test_only_pieces_that_leave_a_state_and_are_long_enough_are_shortened():
    segments = [piece("Survive", LONG, 1), piece("Boss", LONG, 2), piece("Exit", SMALLEST_GAIN_FRAMES, 3)]

    assert shortenable(leg(), segments) == [1]


def test_polishers_keep_their_slots_busy_and_never_take_more():
    pool = RecordingPool()
    improver = Improver("fakes:FakeWrapper", Path("config.yaml"), pool, 2, SECONDS, None)
    segments = [piece("Boss", LONG, 1), piece("Exit", LONG, 2)]

    improver.keep_busy(leg(), segments)
    improver.keep_busy(leg(), segments)

    assert len(pool.jobs) == 2
    assert all(job[3] == LONG - SMALLEST_GAIN_FRAMES for job in pool.jobs)


def test_a_faster_piece_is_handed_over_only_while_the_route_still_holds_it():
    pool = RecordingPool()
    improver = Improver("fakes:FakeWrapper", Path("config.yaml"), pool, 1, SECONDS, None)
    segments = [piece("Boss", LONG, 1)]
    improver.keep_busy(leg(), segments)
    pool.futures[0].set_result(Searched([crossing("Boss", LONG - SMALLEST_GAIN_FRAMES)], None))

    assert improver.found(leg(), [piece("Boss", LONG - 1, 1)]) == []
    improver.keep_busy(leg(), segments)
    pool.futures[1].set_result(Searched([crossing("Boss", LONG - SMALLEST_GAIN_FRAMES)], None))

    assert [(index, found.frames) for index, found in improver.found(leg(), segments)] == [
        (0, LONG - SMALLEST_GAIN_FRAMES)
    ]


def test_the_beam_scout_finds_the_way_out_with_button_patterns(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))

    searched = beam_search(runner, runner.power_on(), SECONDS, None)

    assert searched.crossings[0].end.state == "Boss"
    assert searched.crossings[0].frames == BOSS_FRAMES


def test_the_vision_scout_plays_the_plan_claude_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    runner.emulator.buttons = ["A", "START"]
    prompts: list[str] = []
    monkeypatch.setattr(vision, "asked", lambda prompt, folder: prompts.append(prompt) or 'Plan: [[6, ["A", "START"]]]')

    searched = vision.vision_search(runner, runner.power_on(), "Survive", "- Hold A to jump", None)

    assert [found.end.state for found in searched.crossings] == ["Boss"]
    assert "Buttons: A." in prompts[0] and "- Hold A to jump" in prompts[0]
    assert not any(buttons[1] for buttons in runner.emulator.pressed)


def test_a_failed_claude_call_leaves_the_search_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = SegmentRunner(saving_environment(tmp_path))

    def timed_out(prompt: str, folder: Path) -> str:
        raise subprocess.TimeoutExpired("claude", vision.VISION_SECONDS)

    monkeypatch.setattr(vision, "asked", timed_out)

    assert vision.vision_search(runner, runner.power_on(), "Survive", "- Hold A to jump", None) == Searched([], None)


def test_a_vision_plan_that_is_no_json_list_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(vision, "asked", lambda prompt, folder: "Jump over the van.")

    assert vision.vision_search(runner, runner.power_on(), "Survive", "- Hold A to jump", None) == Searched([], None)


def test_a_repair_is_handed_over_only_while_the_route_still_starts_with_its_original_pieces():
    pool = RecordingPool()
    improver = Improver("fakes:FakeWrapper", Path("config.yaml"), pool, 1, SECONDS, None)
    original = [piece("Boss", LONG, 1), piece("Exit", LONG, 2)]
    improver.repair(Repair("Level 1", original, [piece("Boss", LONG - SMALLEST_GAIN_FRAMES, 5)]), LONG)
    pool.futures[0].set_result(Searched([crossing("Exit", LONG + 1), crossing("Exit", LONG - 1)], None))

    assert improver.repaired(leg(), [piece("Boss", LONG - 1, 1), *original[1:]]) == []
    improver.repair(Repair("Level 1", original, [piece("Boss", LONG - SMALLEST_GAIN_FRAMES, 5)]), LONG)
    pool.futures[1].set_result(Searched([crossing("Exit", LONG + 1), crossing("Exit", LONG - 1)], None))

    assert [found.frames for _, found in improver.repaired(leg(), [*original, piece("Stage", LONG, 3)])] == [LONG - 1]
