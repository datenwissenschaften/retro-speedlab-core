import json
from pathlib import Path

import fakes
import numpy as np
import pytest
from fakes import FakeEmulator, FakeWrapper, fake_environment, write_config

from datenwissenschaften.curriculum import ReverseCurriculum
from datenwissenschaften.environment import factory
from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.recording import active_movie_path
from datenwissenschaften.settings import load_config
from datenwissenschaften.states.landmarks import Landmarks


def test_reset_describes_the_ram_and_asks_the_start_question(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])

    observation, info = env.reset()

    assert json.loads(observation["state"]) == {"lives": 3, "score": 0}
    assert observation["question"] == "Which move survives?"
    assert info["state"] == "Survive"
    assert info["started_from_initial_savestate"] is True
    assert info["episode_bk2_path"].endswith("FakeGame-v0-Level1-000000.bk2")


def test_step_presses_the_mapped_buttons_and_reports_progress(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0), (3, 2)])
    env.reset()

    observation, reward, terminated, truncated, info = env.step(1)

    np.testing.assert_array_equal(env.env.pressed[-1], [0, 1])
    assert reward == 2.0
    assert (terminated, truncated) == (False, False)
    assert info["ram"] == {"lives": 3, "score": 2}
    assert info["won"] is False


def test_transition_changes_the_question_and_counts_as_curriculum_success(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0), (3, 5)])
    env.reset()

    observation, _, terminated, _, info = env.step(0)

    assert info["state_transition"] == ("Survive", "Boss")
    assert observation["question"] == "Which move beats the boss?"
    assert info["curriculum_succeeded"] is True
    assert terminated is True
    assert env.curriculum.curriculum.has_checkpoint("Boss")


def test_winning_ends_the_episode(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0), (3, 9)])
    env.reset()

    _, _, terminated, _, info = env.step(0)

    assert info["won"] is True
    assert info["curriculum_succeeded"] is True
    assert terminated is True


def test_losing_all_lives_records_a_curriculum_failure(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0), (0, 1)])
    env.reset()

    _, _, terminated, _, info = env.step(0)

    assert terminated is True
    assert info["curriculum_succeeded"] is False
    assert env.curriculum.curriculum.typical_steps("Survive") == 1


def test_reset_resumes_from_the_active_curriculum_checkpoint(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    curriculum = env.curriculum.curriculum
    curriculum.save_checkpoint("Boss", b"boss", 7.5)
    for _ in range(ReverseCurriculum.WIN_TARGET):
        curriculum.record_success("Survive", 10)

    observation, info = env.reset()

    assert observation["question"] == "Which move beats the boss?"
    assert info["started_from_initial_savestate"] is False
    assert info["episode_start_state"] == "Boss"
    assert info["episode_start_score"] == 7.5
    assert env.env.pressed[-1].tolist() == [0, 0]


def test_a_new_checkpoint_remembers_the_score_that_reached_it(tmp_path: Path):
    run = CurriculumRun(tmp_path, ("Survive", "Boss"))
    run.begin_episode()
    run.add_reward(3.0, False)

    run.transition("Survive", "Boss", b"boss", 2.0)

    assert run.curriculum.entry_score("Boss") == 5.0


def test_unknown_curriculum_state_fails_fast(tmp_path: Path):
    with pytest.raises(ValueError, match="Unknown state"):
        fake_environment(tmp_path, [(3, 0)])._state_class("Missing")


def test_every_action_needs_a_description(tmp_path: Path):
    class Undescribed(FakeWrapper):
        action_descriptions = {"left": "move left"}

    with pytest.raises(ValueError, match="description"):
        Undescribed(
            FakeEmulator(tmp_path, [(3, 0)]),
            CurriculumRun(tmp_path, ("Survive",)),
            Landmarks(tmp_path / "landmarks.json"),
            "Level1",
        )


def test_actions_must_be_button_sequences(tmp_path: Path):
    class Flat(FakeWrapper):
        action_table = np.array([[1, 0], [0, 1]], dtype=np.int8)

    with pytest.raises(ValueError, match="frames"):
        Flat(
            FakeEmulator(tmp_path, [(3, 0)]),
            CurriculumRun(tmp_path, ("Survive",)),
            Landmarks(tmp_path / "landmarks.json"),
            "Level1",
        )


def test_training_memory_reset_rebuilds_the_curriculum_and_forgets_landmarks(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    previous = env.curriculum.curriculum
    env.state_machine.landmarks.remember("door", (3, 4))

    env.reset_training_memory()

    assert env.curriculum.curriculum is not previous
    assert env.state_machine.landmarks.recall("door") is None


def test_recording_is_mandatory(tmp_path: Path):
    with pytest.raises(RuntimeError, match="recording"):
        active_movie_path(FakeEmulator(tmp_path, [(3, 0)]))


def test_factory_records_into_the_game_and_savestate_folder(tmp_path: Path, monkeypatch):
    calls = {}
    monkeypatch.setattr(factory, "import_roms", lambda roms: calls.setdefault("roms", roms))

    def make(game, state, render_mode, record):
        calls.update(game=game, state=state, record=record)
        return FakeEmulator(Path(record), [(3, 0)])

    monkeypatch.setattr(factory.retro, "make", make)
    config = load_config(write_config(tmp_path))

    env = factory.make_environment(FakeWrapper, config)

    assert calls["roms"] == config.paths.roms_path
    assert (calls["game"], calls["state"]) == ("FakeGame-v0", "Level1")
    assert calls["record"] == str(config.paths.record_dir / "FakeGame-v0" / "Level1")
    assert env.curriculum.state_names == ("Survive", "Boss")


def test_what_a_state_sees_becomes_part_of_layas_text(tmp_path: Path):
    from datenwissenschaften.vision.detection import Detection

    door = Detection("door", 1, 1, 2, 2)

    class Seeing(fakes.Survive):
        def describe(self):
            return {"door": {"visible": True, "direction": "up"}}

        def detections(self):
            return (door,)

    class SeeingWrapper(FakeWrapper):
        start_state_cls = Seeing
        state_classes = (Seeing, fakes.Boss)

    env = SeeingWrapper(
        FakeEmulator(tmp_path, [(3, 0), (3, 1)]),
        CurriculumRun(tmp_path, ("Seeing", "Boss")),
        Landmarks(tmp_path / "landmarks.json"),
        "Level1",
    )

    observation, info = env.reset()
    _, _, _, _, step_info = env.step(0)

    assert json.loads(observation["state"]) == {"lives": 3, "score": 0, "door": {"visible": True, "direction": "up"}}
    assert info["detections"] == (door,)
    assert step_info["detections"] == (door,)
