import json
from pathlib import Path

import pytest

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment import level_clock as level_clock_module
from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.level_clock import LevelClock
from datenwissenschaften.environment.level_drill import MIN_ENDINGS, RELIABLE_PASSES
from datenwissenschaften.environment.levels import LevelTargets, curriculum_targets, level_map
from datenwissenschaften.states.state import State

FRAME_RATE = 60.0
ATTEMPTS = 20


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
    return level_run(tmp_path, monkeypatch, True)


def level_run(tmp_path: Path, monkeypatch, beats_game: bool) -> CurriculumRun:
    monkeypatch.setattr(level_clock_module, "publish_metadata", lambda section, values, replace: None)
    levels = level_map(LEVELS, STATES)
    targets = LevelTargets(curriculum_targets(STATES, levels), levels, beats_game)
    curriculum_run = CurriculumRun(
        tmp_path / "curriculum",
        targets,
        "PowerOn",
        tmp_path / "seeds",
        LevelClock(tmp_path / "times.json", FRAME_RATE),
        True,
    )
    curriculum_run.curriculum.save_checkpoint("Eat", b"eat", 0.0)
    curriculum_run.curriculum.save_checkpoint("Door", b"door", 0.0)
    for state in ("Menu", "Eat", "Door"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            curriculum_run.curriculum.record_success(state, 1)
    return curriculum_run


def play_frames(curriculum_run: CurriculumRun, frames: int) -> None:
    for _ in range(frames):
        curriculum_run.tally.count_step()


def test_a_mastered_level_is_practised_whole_from_its_first_state(run: CurriculumRun):
    assert run.begin_episode() == "Eat"
    assert run.start_state == "Level 1"


def test_only_leaving_the_level_beats_it_and_records_its_time(run: CurriculumRun, tmp_path: Path):
    run.begin_episode()
    play_frames(run, 600)
    assert run.transition("Eat", "Door", b"door", 0.0) == (False, False)
    play_frames(run, 300)

    assert run.transition("Door", "Next", b"next", 0.0) == (True, True)
    assert run.finish_step(False, False, 0.0, ("Door", "Next"), (True, True), "Next")["curriculum_state"] == "Level 1"
    assert run.curriculum.wins("Level 1") == 1
    assert json.loads((tmp_path / "times.json").read_text()) == {
        "Level 1": {"best_seconds": 15.0, "last_seconds": 15.0, "runs": 1}
    }


def test_a_win_inside_the_level_beats_it_too(run: CurriculumRun):
    run.begin_episode()
    play_frames(run, 900)

    assert run.finish_step(True, True, 0.0, None, (False, False), "Door")["curriculum_succeeded"] is True
    assert run.clock.times()["Level 1"] == {"best_seconds": 15.0, "last_seconds": 15.0, "runs": 1}


def test_the_first_level_win_masters_the_level_and_moves_on(run: CurriculumRun):
    run.begin_episode()
    play_frames(run, 60)
    run.finish_step(True, True, 0.0, None, (False, False), "Door")

    assert run.curriculum.is_mastered("Level 1")
    assert run.curriculum.active_state() == "Next"


def test_a_level_run_from_power_on_times_the_level_from_its_first_state(run: CurriculumRun, tmp_path: Path):
    run.curriculum._checkpoint_path("Eat").unlink()
    assert run.begin_episode() is None
    play_frames(run, 648)
    run.transition("Menu", "Eat", b"eat", 0.0)
    play_frames(run, 600)

    assert run.transition("Door", "Next", b"next", 0.0) == (True, True)
    assert run.clock.times()["Level 1"]["last_seconds"] == 10.0


def test_a_curriculum_that_stops_before_the_end_of_the_game_keeps_practising_its_last_target(
    tmp_path: Path, monkeypatch
):
    run = level_run(tmp_path, monkeypatch, False)
    run.curriculum.save_checkpoint("Next", b"next", 0.0)
    for target in ("Level 1", "Next"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            run.curriculum.record_success(target, 1)

    assert run.begin_episode() == "Next"
    assert run.finish_step(False, True, 0.0, None, (False, False), "Door")["curriculum_state"] == "Next"


def test_once_every_target_of_a_curriculum_that_beats_the_game_is_mastered_attempts_are_full_runs(
    run: CurriculumRun,
):
    for target in ("Level 1", "Next"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            run.curriculum.record_success(target, 1)

    assert run.begin_episode() is None
    assert run.finish_step(False, True, 0.0, None, (False, False), "Door")["curriculum_state"] == "Full run"
    assert run.transition("Door", "Next", b"next", 0.0) == (False, False)


def test_a_beaten_level_is_never_practised_again(tmp_path: Path, monkeypatch):
    run = level_run(tmp_path, monkeypatch, False)
    for _ in range(ReverseCurriculum.WIN_TARGET):
        run.curriculum.record_success("Level 1", 1)

    targets = set()
    for _ in range(ATTEMPTS):
        run.begin_episode()
        targets.add(run.start_state)

    assert targets == {"Next"}


def fail_level_attempts(run: CurriculumRun, endings: list[str]) -> None:
    for state in endings:
        run.begin_episode()
        run.finish_step(False, True, 0.0, None, (False, False), state)


def test_a_level_drills_the_state_where_most_of_its_attempts_end(run: CurriculumRun):
    fail_level_attempts(run, ["Door"] * (MIN_ENDINGS - 2) + ["Eat"])
    assert run.begin_episode() == "Eat"
    run.finish_step(False, True, 0.0, None, (False, False), "Door")

    assert run.begin_episode() == "Door"
    assert run.start_state == "Door"
    assert run.drill.progress("Level 1")["endings"] == {"Door": MIN_ENDINGS - 1, "Eat": 1}


def test_a_drill_ends_once_the_state_passes_reliably_and_mastery_stays(run: CurriculumRun):
    fail_level_attempts(run, ["Door"] * MIN_ENDINGS)
    for _ in range(RELIABLE_PASSES - 1):
        run.begin_episode()
        assert run.transition("Door", "Next", b"next", 0.0) == (True, False)
    run.begin_episode()
    run.finish_step(False, True, 0.0, None, (False, False), "Door")
    assert run.drill.progress("Level 1")["drill_attempts"] == RELIABLE_PASSES

    run.begin_episode()
    run.transition("Door", "Next", b"next", 0.0)

    assert run.begin_episode() == "Eat"
    assert run.start_state == "Level 1"
    assert run.curriculum.is_mastered("Door")
    assert run.drill.progress("Level 1") == {"endings": {}, "drill": None, "drill_passes": 0, "drill_attempts": 0}


def test_practice_emulators_never_count_level_endings(run: CurriculumRun):
    run.counts_outcomes = False
    fail_level_attempts(run, ["Door"] * MIN_ENDINGS)

    assert run.drill.target("Level 1") is None
