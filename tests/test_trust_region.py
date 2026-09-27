from datenwissenschaften.laya.trust_region import (
    INITIAL_ENTROPY_COEFFICIENT,
    MIN_LEARNING_RATE_SCALE,
    TARGET_KL,
    TrustRegion,
)

ACTIONS = 3


def test_learning_rate_shrinks_after_a_step_that_moved_the_policy_too_far():
    region = TrustRegion()

    region.adapt(TARGET_KL * 10, 0.8, ACTIONS)

    assert region.learning_rate_scale == 0.5


def test_learning_rate_grows_after_a_timid_step_and_stays_within_bounds():
    region = TrustRegion()
    region.adapt(0.0, 0.8, ACTIONS)
    grown = region.learning_rate_scale

    for _ in range(100):
        region.adapt(TARGET_KL * 10, 0.8, ACTIONS)

    assert grown > 1.0
    assert region.learning_rate_scale == MIN_LEARNING_RATE_SCALE


def test_learning_rate_is_kept_inside_the_trust_region():
    region = TrustRegion()

    region.adapt(TARGET_KL, 0.8, ACTIONS)

    assert region.learning_rate_scale == 1.0


def test_entropy_bonus_grows_when_laya_becomes_too_certain_and_relaxes_again():
    region = TrustRegion()

    region.adapt(TARGET_KL, 0.0, ACTIONS)
    grown = region.entropy_coefficient
    region.adapt(TARGET_KL, 1.0, ACTIONS)

    assert grown > INITIAL_ENTROPY_COEFFICIENT
    assert region.entropy_coefficient < grown
