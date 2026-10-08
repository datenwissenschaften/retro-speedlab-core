from dataclasses import dataclass

from loguru import logger

STALL_DECISIONS = 200_000
STALLED_ENTROPY_SHARE = 0.9
STALLED_EXPLAINED_VARIANCE = 0.1
SMOOTHING = 0.1


@dataclass(slots=True)
class Trend:
    entropy_share: float
    explained_variance: float

    def blend(self, entropy_share: float, explained_variance: float) -> None:
        self.entropy_share += SMOOTHING * (entropy_share - self.entropy_share)
        self.explained_variance += SMOOTHING * (explained_variance - self.explained_variance)

    @property
    def flat(self) -> bool:
        return self.entropy_share >= STALLED_ENTROPY_SHARE and self.explained_variance < STALLED_EXPLAINED_VARIANCE


class StallWatch:
    def __init__(self) -> None:
        self.trends: dict[str, Trend] = {}
        self.stalled: set[str] = set()

    def observe(self, state: str, decisions: int, entropy_share: float, explained_variance: float) -> bool:
        if state in self.trends:
            self.trends[state].blend(entropy_share, explained_variance)
        else:
            self.trends[state] = Trend(entropy_share, explained_variance)
        trend = self.trends[state]
        stalled = decisions >= STALL_DECISIONS and trend.flat
        if stalled and state not in self.stalled:
            logger.warning(
                f"Laya is stalled in {state}: {decisions:,} decisions, entropy share {trend.entropy_share:.2f}, "
                f"explained variance {trend.explained_variance:.2f}; its facts or rewards need work"
            )
        if not stalled and state in self.stalled:
            logger.info(f"Laya is learning again in {state}")
        self.stalled = self.stalled | {state} if stalled else self.stalled - {state}
        return stalled
