import math

TARGET_KL = 0.01
KL_TOLERANCE = 2.0
LEARNING_RATE_DECREASE = 0.5
STRONGEST_DECREASE = 0.1
STRONGEST_INCREASE = 10.0
KL_FLOOR = 1e-12
MIN_LEARNING_RATE_SCALE = 1e-2
MAX_LEARNING_RATE_SCALE = 1e6


class TrustRegion:
    def __init__(self) -> None:
        self.learning_rate_scale = 1.0

    def state_dict(self) -> dict[str, float]:
        return {"learning_rate_scale": self.learning_rate_scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.learning_rate_scale = state["learning_rate_scale"]

    def accepts(self, kl: float) -> bool:
        return kl <= TARGET_KL * KL_TOLERANCE

    def backtrack_fraction(self, kl: float) -> float:
        return math.sqrt(TARGET_KL / kl)

    def adapt(self, kl: float) -> None:
        if kl > TARGET_KL * KL_TOLERANCE:
            self.learning_rate_scale *= max(STRONGEST_DECREASE, min(LEARNING_RATE_DECREASE, TARGET_KL / kl))
        elif kl < TARGET_KL / KL_TOLERANCE:
            self.learning_rate_scale *= min(STRONGEST_INCREASE, math.sqrt(TARGET_KL / max(kl, KL_FLOOR)))
        self.learning_rate_scale = min(MAX_LEARNING_RATE_SCALE, max(MIN_LEARNING_RATE_SCALE, self.learning_rate_scale))
