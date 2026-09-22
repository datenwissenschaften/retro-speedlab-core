import pytest

from datenwissenschaften.curriculum import ReverseCurriculum


def test_save_checkpoint_is_a_no_op_once_mastered(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    for _ in range(64):
        curriculum.record_success("Start", 4)

    assert curriculum.save_checkpoint("Start", b"ignored") is False


def test_save_checkpoint_does_not_overwrite_an_existing_checkpoint(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.save_checkpoint("Finish", b"first") is True
    assert curriculum.save_checkpoint("Finish", b"second") is False
    assert curriculum.checkpoint("Finish") == b"first"


def test_episode_start_state_is_none_once_the_curriculum_is_complete(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    for state_name in ("Start", "Finish"):
        for _ in range(64):
            curriculum.record_success(state_name, 4)

    assert curriculum.episode_start_state() is None


def test_episode_start_state_is_none_without_any_earlier_checkpoint(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    for _ in range(64):
        curriculum.record_success("Start", 4)

    assert curriculum.active_state() == "Middle"
    assert curriculum.episode_start_state() is None


def test_record_success_is_a_no_op_once_mastered(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    for _ in range(64):
        curriculum.record_success("Start", 4)

    assert curriculum.record_success("Start", 4) is False


def test_record_failure_is_a_no_op_once_mastered(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    for _ in range(64):
        curriculum.record_success("Start", 4)

    assert curriculum.record_failure("Start", 4, 1.0) is False


def test_record_failure_rejects_non_finite_scores(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"state")

    with pytest.raises(ValueError, match="score must be finite"):
        curriculum.record_failure("Finish", 4, float("nan"))


def test_unknown_state_name_is_rejected(tmp_path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    with pytest.raises(ValueError, match="Unknown curriculum state"):
        curriculum.has_checkpoint("Nowhere")
