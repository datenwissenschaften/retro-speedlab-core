from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from fakes import saving_environment

from datenwissenschaften.speedrun import vision
from datenwissenschaften.speedrun.beam import beam_search
from datenwissenschaften.speedrun.improver import SMALLEST_GAIN_FRAMES, Improver, shortenable
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
    improver = Improver("fakes:FakeWrapper", Path("config.yaml"), pool, 2, SECONDS)
    segments = [piece("Boss", LONG, 1), piece("Exit", LONG, 2)]

    improver.keep_busy(leg(), segments)
    improver.keep_busy(leg(), segments)

    assert len(pool.jobs) == 2
    assert all(job[3] == LONG - SMALLEST_GAIN_FRAMES for job in pool.jobs)


def test_a_faster_piece_is_handed_over_only_while_the_route_still_holds_it():
    pool = RecordingPool()
    improver = Improver("fakes:FakeWrapper", Path("config.yaml"), pool, 1, SECONDS)
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
    monkeypatch.setattr(vision, "asked", lambda prompt, folder: 'Plan: [[6, ["A"]]]')

    searched = vision.vision_search(runner, runner.power_on(), "Survive")

    assert [found.end.state for found in searched.crossings] == ["Boss"]


def test_a_vision_plan_that_is_no_json_list_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(vision, "asked", lambda prompt, folder: "Jump over the van.")

    assert vision.vision_search(runner, runner.power_on(), "Survive") == Searched([], None)
