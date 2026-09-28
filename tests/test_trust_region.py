import pytest

from datenwissenschaften.laya.trust_region import (
    MAX_LEARNING_RATE_SCALE,
    MIN_LEARNING_RATE_SCALE,
    STRONGEST_INCREASE,
    TARGET_KL,
    TrustRegion,
)


def test_learning_rate_shrinks_in_proportion_to_the_overshoot():
    mild, severe, extreme = TrustRegion(), TrustRegion(), TrustRegion()

    mild.adapt(TARGET_KL * 2.5)
    severe.adapt(TARGET_KL * 5)
    extreme.adapt(TARGET_KL * 60)

    assert mild.learning_rate_scale == pytest.approx(0.4)
    assert severe.learning_rate_scale == pytest.approx(0.2)
    assert extreme.learning_rate_scale == pytest.approx(0.1)


def test_learning_rate_grows_after_a_timid_step_and_stays_within_bounds():
    region = TrustRegion()
    region.adapt(0.0)
    grown = region.learning_rate_scale

    for _ in range(100):
        region.adapt(TARGET_KL * 10)

    assert grown > 1.0
    assert region.learning_rate_scale == MIN_LEARNING_RATE_SCALE


def test_learning_rate_is_kept_inside_the_trust_region():
    region = TrustRegion()

    region.adapt(TARGET_KL)

    assert region.learning_rate_scale == 1.0


def test_learning_rate_grows_toward_the_target_in_proportion_to_the_shortfall():
    timid, frozen = TrustRegion(), TrustRegion()

    timid.adapt(TARGET_KL / 4)
    frozen.adapt(0.0)

    assert timid.learning_rate_scale == pytest.approx(2.0)
    assert frozen.learning_rate_scale == STRONGEST_INCREASE


def test_a_saturated_policy_cannot_push_the_learning_rate_past_the_ceiling():
    saturated, restored = TrustRegion(), TrustRegion()

    for _ in range(100):
        saturated.adapt(0.0)
    restored.load_state_dict({"learning_rate_scale": 350_440.0})

    assert saturated.learning_rate_scale == MAX_LEARNING_RATE_SCALE
    assert restored.learning_rate_scale == MAX_LEARNING_RATE_SCALE
