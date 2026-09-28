import math

TARGET_KL = 0.01
KL_TOLERANCE = 2.0
LEARNING_RATE_DECREASE = 0.5
STRONGEST_DECREASE = 0.1
LEARNING_RATE_INCREASE = 1.2
MIN_LEARNING_RATE_SCALE = 1e-3
MAX_LEARNING_RATE_SCALE = 10.0
INITIAL_ENTROPY_COEFFICIENT = 0.05
MIN_ENTROPY_COEFFICIENT = 1e-3
MAX_ENTROPY_COEFFICIENT = 1.0
TARGET_ENTROPY_FRACTION = 0.5
ENTROPY_ADAPTATION_RATE = 0.5


class TrustRegion:
    def __init__(self) -> None:
        self.learning_rate_scale = 1.0
        self.entropy_coefficient = INITIAL_ENTROPY_COEFFICIENT

    def adapt(self, kl: float, entropy: float, action_count: int) -> None:
        self._adapt_learning_rate(kl)
        self._adapt_entropy_coefficient(entropy, action_count)

    def state_dict(self) -> dict[str, float]:
        return {"learning_rate_scale": self.learning_rate_scale, "entropy_coefficient": self.entropy_coefficient}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.learning_rate_scale = state["learning_rate_scale"]
        self.entropy_coefficient = state["entropy_coefficient"]

    def _adapt_learning_rate(self, kl: float) -> None:
        if kl > TARGET_KL * KL_TOLERANCE:
            self.learning_rate_scale *= max(STRONGEST_DECREASE, min(LEARNING_RATE_DECREASE, TARGET_KL / kl))
        elif kl < TARGET_KL / KL_TOLERANCE:
            self.learning_rate_scale *= LEARNING_RATE_INCREASE
        self.learning_rate_scale = min(MAX_LEARNING_RATE_SCALE, max(MIN_LEARNING_RATE_SCALE, self.learning_rate_scale))

    def _adapt_entropy_coefficient(self, entropy: float, action_count: int) -> None:
        target = TARGET_ENTROPY_FRACTION * math.log(action_count)
        adapted = self.entropy_coefficient * math.exp(ENTROPY_ADAPTATION_RATE * (target - entropy) / target)
        self.entropy_coefficient = min(MAX_ENTROPY_COEFFICIENT, max(MIN_ENTROPY_COEFFICIENT, adapted))
