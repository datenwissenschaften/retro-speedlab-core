import torch

from datenwissenschaften.probe.readability import MIN_OFFSET_R2, Readability, held_out_r2, numeric_facts
from datenwissenschaften.probe.samples import Sample
from datenwissenschaften.states.facts import Offset

SAMPLES = 400


def test_offsets_are_required_numbers_and_other_numbers_are_only_reported():
    sample = Sample("Play", {"lives": 3, "alive": True, "target": Offset(4, -2), "name": "door"}, "", "")

    assert numeric_facts(sample) == {"lives": (3.0, False), "target.right": (4.0, True), "target.down": (-2.0, True)}


def test_a_fact_written_into_the_features_reads_back_and_noise_does_not():
    generator = torch.Generator().manual_seed(1)
    fact = torch.randn(SAMPLES, generator=generator, dtype=torch.float64)
    noise = torch.randn(SAMPLES, 8, generator=generator, dtype=torch.float64)

    readable = held_out_r2(torch.cat([fact[:, None], noise], -1), fact)
    unreadable = held_out_r2(noise, fact)

    assert readable > MIN_OFFSET_R2 > unreadable


def test_only_required_facts_can_fail_the_probe():
    assert not Readability("Play", "target.right", MIN_OFFSET_R2 - 0.1, True).passed
    assert Readability("Play", "score", 0.0, False).passed
