import random
from dataclasses import dataclass

from datenwissenschaften.policy.prior import PolicyPrior
from datenwissenschaften.tas.prior import ActionPrior
from datenwissenschaften.tas.runner import SegmentRunner

TABLE_SHARE = 0.3
POLICY_SHARE = 0.2


@dataclass(slots=True, frozen=True)
class Preferences:
    table: ActionPrior
    policy: PolicyPrior

    def choose(self, runner: SegmentRunner, actions: int, rng: random.Random) -> int | None:
        roll = rng.random()
        if roll < TABLE_SHARE:
            return self.table.choose(runner.context(), actions, rng)
        if roll < TABLE_SHARE + POLICY_SHARE:
            return self.policy.choose(runner.memory(), actions, rng)
        return None
