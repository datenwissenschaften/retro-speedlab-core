import json
from pathlib import Path

import pytest

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment import level_clock as level_clock_module
from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.level_clock import LevelClock
from datenwissenschaften.environment.levels import LevelTargets, curriculum_targets, level_map
from datenwissenschaften.states.state import State

FRAME_RATE = 60.0


class Menu(State):
    pass


class Eat(State):
    pass


class Door(State):
    pass


class Next(State):
    pass


STATES = (Menu, Eat, Door, Next)
LEVELS = (("Level 1", (Eat, Door)),)


def test_a_level_becomes_a_target_right_after_its_last_state():
    levels = level_map(LEVELS, STATES)

    assert levels == {"Level 1": ("Eat", "Door")}
    assert curriculum_targets(STATES, levels) == ("Menu", "Eat", "Door", "Level 1", "Next")


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ((("Eat", (Door,)),), "name of a state"),
        ((("Level 1", (Eat, Menu)),), "follow each other"),
        ((("Level 1", ()),), "must be one of"),
    ],
)
def test_levels_must_name_neighbouring_states(levels, message):
    with pytest.raises(ValueError, match=message):
        level_map(levels, STATES)


@pytest.fixture
def run(tmp_path: Path, monkeypatch) -> CurriculumRun:
    monkeypatch.setattr(level_clock_module, "publish_metadata", lambda section, values, replace: None)
    levels = level_map(LEVELS, STATES)
    targets = LevelTargets(curriculum_targets(STATES, levels), levels)
    curriculum_run = CurriculumRun(
        tmp_path / "curriculum", targets, "PowerOn", tmp_path / "seeds", LevelClock(tmp_path / "times.json", FRAME_RATE)
    )
    curriculum_run.curriculum.save_checkpoint("Eat", b"eat", 0.0)
    curriculum_run.curriculum.save_checkpoint("Door", b"door", 0.0)
    for state in ("Menu", "Eat", "Door"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            curriculum_run.curriculum.record_success(state, 1)
    return curriculum_run


def play_frames(curriculum_run: CurriculumRun, frames: int) -> None:
    for _ in range(frames):
        curriculum_run.count_step()


def test_a_mastered_level_is_practised_whole_from_its_first_state(run: CurriculumRun):
    assert run.begin_episode() == "Eat"
    assert run.start_state == "Level 1"


def test_only_leaving_the_level_beats_it_and_records_its_time(run: CurriculumRun, tmp_path: Path):
    run.begin_episode()
    play_frames(run, 600)
    assert run.transition("Eat", "Door", b"door", 0.0) == (False, False)
    play_frames(run, 300)

    assert run.transition("Door", "Next", b"next", 0.0) == (True, False)
    assert run.finish_step(False, False, 0.0, ("Door", "Next"), (True, False))["curriculum_state"] == "Level 1"
    assert run.curriculum.wins("Level 1") == 1
    assert json.loads((tmp_path / "times.json").read_text()) == {
        "Level 1": {"best_seconds": 15.0, "last_seconds": 15.0, "runs": 1}
    }


def test_a_win_inside_the_level_beats_it_too_and_keeps_the_best_time(run: CurriculumRun, tmp_path: Path):
    for frames in (1200, 900, 1500):
        run.begin_episode()
        play_frames(run, frames)
        outcome = run.finish_step(True, True, 0.0, None, (False, False))
        assert outcome["curriculum_succeeded"] is True

    assert run.clock.times()["Level 1"] == {"best_seconds": 15.0, "last_seconds": 25.0, "runs": 3}


def test_eight_level_wins_master_the_level_and_move_on(run: CurriculumRun):
    for _ in range(ReverseCurriculum.WIN_TARGET):
        run.begin_episode()
        play_frames(run, 60)
        run.finish_step(True, True, 0.0, None, (False, False))

    assert run.curriculum.is_mastered("Level 1")
    assert run.curriculum.active_state() == "Next"


def test_a_level_run_from_power_on_times_the_level_from_its_first_state(run: CurriculumRun, tmp_path: Path):
    run.curriculum._checkpoint_path("Eat").unlink()
    assert run.begin_episode() is None
    play_frames(run, 648)
    run.transition("Menu", "Eat", b"eat", 0.0)
    play_frames(run, 600)

    assert run.transition("Door", "Next", b"next", 0.0) == (True, False)
    assert run.clock.times()["Level 1"]["last_seconds"] == 10.0


def test_once_every_target_is_mastered_attempts_are_full_runs(run: CurriculumRun):
    for target in ("Level 1", "Next"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            run.curriculum.record_success(target, 1)

    assert run.begin_episode() is None
    assert run.finish_step(False, True, 0.0, None, (False, False))["curriculum_state"] == "Full run"
    assert run.transition("Door", "Next", b"next", 0.0) == (False, False)
