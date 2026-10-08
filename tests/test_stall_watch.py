from datenwissenschaften.training.stall_watch import STALL_DECISIONS, StallWatch


def test_a_state_stalls_only_after_enough_decisions_without_learning():
    watch = StallWatch()

    early = watch.observe("Play", STALL_DECISIONS - 1, 0.99, 0.0)
    late = watch.observe("Play", STALL_DECISIONS, 0.99, 0.0)

    assert (early, late) == (False, True)
    assert watch.stalled == {"Play"}


def test_a_learning_state_is_never_stalled_and_a_recovered_one_is_released():
    watch = StallWatch()
    watch.observe("Play", STALL_DECISIONS, 0.99, 0.0)

    for _ in range(40):
        stalled = watch.observe("Play", STALL_DECISIONS, 0.5, 0.6)

    assert not stalled
    assert watch.stalled == set()
    assert not watch.observe("Heavy", STALL_DECISIONS, 0.5, 0.0)
