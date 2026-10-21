from dataclasses import dataclass

import numpy as np
from scipy.stats import bootstrap, mannwhitneyu

CONFIDENCE = 0.95
RESAMPLES = 2000
SEED = 0


@dataclass(slots=True, frozen=True)
class Comparison:
    before: float
    after: float
    difference_low: float
    difference_high: float
    probability_of_improvement: float

    def metrics(self, name: str) -> dict[str, float]:
        return {
            f"{name}/before": self.before,
            f"{name}/after": self.after,
            f"{name}/difference_low": self.difference_low,
            f"{name}/difference_high": self.difference_high,
            f"{name}/probability_of_improvement": self.probability_of_improvement,
        }


def mean_difference(after: np.ndarray, before: np.ndarray, axis: int) -> np.ndarray:
    return np.mean(after, axis=axis) - np.mean(before, axis=axis)


def compare(before: list[float], after: list[float]) -> Comparison:
    if min(len(before), len(after)) < 2:
        raise ValueError("An A/B comparison needs at least two attempts per model.")
    interval = bootstrap(
        (after, before),
        mean_difference,
        n_resamples=RESAMPLES,
        confidence_level=CONFIDENCE,
        method="percentile",
        rng=np.random.default_rng(SEED),
    ).confidence_interval
    improvement = mannwhitneyu(after, before, alternative="greater").statistic / (len(after) * len(before))
    return Comparison(
        before=float(np.mean(before)),
        after=float(np.mean(after)),
        difference_low=float(interval.low),
        difference_high=float(interval.high),
        probability_of_improvement=float(improvement),
    )
