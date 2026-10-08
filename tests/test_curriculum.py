from pathlib import Path

from datenwissenschaften.curriculum import ReverseCurriculum


def test_curriculum_masters_states_in_order_even_with_deeper_checkpoints(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    curriculum.save_checkpoint("Middle", b"middle", 0.0)
    curriculum.save_checkpoint("Finish", b"finish", 0.0)

    assert curriculum.active_state() == "Start"
    for _ in range(ReverseCurriculum.WIN_TARGET - 1):
        assert curriculum.record_success("Start", 4) is False
    assert curriculum.record_success("Start", 4) is True
    assert curriculum.active_state() == "Middle"
    assert curriculum.checkpoint("Middle") == b"middle"


def test_mastered_state_is_never_selected_again_when_checkpoint_is_missing(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle"))
    for _ in range(ReverseCurriculum.WIN_TARGET):
        curriculum.record_success("Start", 4)

    assert curriculum.active_state() == "Middle"
    assert curriculum.has_checkpoint("Middle") is False


def test_bad_checkpoint_recovers_from_nearest_mastered_checkpoint(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    curriculum.save_checkpoint("Middle", b"middle", 0.0)
    for state_name in ("Start", "Middle"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            curriculum.record_success(state_name, 4)
    curriculum.save_checkpoint("Finish", b"bad-finish", 0.0)

    curriculum._checkpoint_path("Finish").unlink()

    assert curriculum.active_state() == "Finish"
    assert curriculum.episode_start_state() == "Middle"
    assert curriculum.is_mastered("Middle") is True


def test_mastered_checkpoint_is_not_used_when_target_checkpoint_is_healthy(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Middle", "Finish"))
    curriculum.save_checkpoint("Middle", b"middle", 0.0)
    curriculum.save_checkpoint("Finish", b"finish", 0.0)
    for state_name in ("Start", "Middle"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            curriculum.record_success(state_name, 4)

    assert curriculum.active_state() == "Finish"
    assert curriculum.episode_start_state() == "Finish"


def test_curriculum_counts_wins_across_failures(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    curriculum.record_success("Finish", 16)
    curriculum.record_success("Finish", 16)
    curriculum.record_failure("Finish", 16, 10.0)

    assert curriculum.wins("Finish") == 2
    assert curriculum.record_success("Finish", 16) is False
    assert curriculum.wins("Finish") == 3


def test_initial_state_only_completes_after_success_threshold(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    for _ in range(ReverseCurriculum.WIN_TARGET - 1):
        assert curriculum.record_success("Start", 1) is False
    assert curriculum.record_success("Start", 1) is True
    assert curriculum.progress()["Start"]["mastered"] is True


def test_bad_checkpoint_is_deleted_after_persistent_score_stagnation(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"unrecoverable", 0.0)

    assert curriculum.record_failure("Finish", 100, 10.0) is False
    for _ in range(127):
        assert curriculum.record_failure("Finish", 100, 10.0) is False
    assert curriculum.record_failure("Finish", 100, 10.0) is True
    assert curriculum.active_state() == "Start"
    assert curriculum.has_checkpoint("Finish") is False
    assert curriculum.stagnation_evidence("Finish") == 0


def test_completed_curriculum_has_no_active_checkpoint_stage(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    for state_name in ("Start", "Finish"):
        for _ in range(ReverseCurriculum.WIN_TARGET):
            curriculum.record_success(state_name, 4)

    assert curriculum.active_state() is None
    assert not any(state["active"] for state in curriculum.progress().values())


def test_score_improvement_resets_stagnation_evidence(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"recoverable", 0.0)

    curriculum.record_failure("Finish", 100, 10.0)
    curriculum.record_failure("Finish", 100, 10.0)
    assert curriculum.stagnation_evidence("Finish") == 1

    curriculum.record_failure("Finish", 100, 11.0)
    assert curriculum.stagnation_evidence("Finish") == 0
    assert curriculum.best_score("Finish") == 11.0


def test_bad_checkpoint_evidence_target_allows_extended_stagnation(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.bad_checkpoint_evidence_target("Finish") == 128
    assert curriculum.progress()["Finish"]["bad_checkpoint_evidence_target"] == 128


def test_declining_scores_accumulate_evidence_twice_as_fast(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.save_checkpoint("Finish", b"declining", 0.0)

    curriculum.record_failure("Finish", 100, 10.0)
    curriculum.record_failure("Finish", 100, 9.0)
    assert curriculum.stagnation_evidence("Finish") == 2
    curriculum.record_failure("Finish", 100, 8.0)
    assert curriculum.stagnation_evidence("Finish") == 4


def test_bad_checkpoint_detector_never_deletes_initial_state(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.record_failure("Start", 1, 0.0) is False
    assert curriculum.stagnation_evidence("Start") == 0


def test_success_target_is_the_same_for_every_state(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))

    assert curriculum.win_target("Finish") == ReverseCurriculum.WIN_TARGET
    curriculum.record_success("Finish", 256)
    assert curriculum.win_target("Finish") == ReverseCurriculum.WIN_TARGET
    assert curriculum.progress()["Finish"]["wins"] == 1
    assert curriculum.progress()["Finish"]["win_target"] == ReverseCurriculum.WIN_TARGET


def test_a_win_only_counts_within_the_speed_margin_of_the_median_recent_win(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.record_success("Start", 80)
    curriculum.record_success("Start", 100)

    assert curriculum.step_limit("Start") == 112
    assert curriculum.record_success("Start", 400) is False
    assert curriculum.wins("Start") == 2
    assert curriculum.step_limit("Start") == 125
    curriculum.record_success("Start", 120)
    assert curriculum.wins("Start") == 3
    assert curriculum.progress()["Start"]["win_step_limit"] == 137


def test_one_lucky_fast_win_never_blocks_mastery(tmp_path: Path):
    curriculum = ReverseCurriculum(tmp_path, ("Start", "Finish"))
    curriculum.record_success("Start", 50)

    while not curriculum.is_mastered("Start"):
        curriculum.record_success("Start", 4000)

    assert curriculum.recent_win_steps("Start")[-1] == 4000
