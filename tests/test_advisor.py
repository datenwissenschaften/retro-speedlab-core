import time
from pathlib import Path

import numpy as np
import torch

from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.advisor.coach import ROLLOUT_STEPS, Coach
from datenwissenschaften.advisor.inputs import FACT_SLOTS, RAM_SCALE, encode
from datenwissenschaften.rollout import Rollout
from datenwissenschaften.states.facts import Offset
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.practice import PracticeStep

ACTIONS = ("left", "right")
INPUTS = np.ones(4, dtype=np.float32)
UPDATES = 40
COACH_WAIT_SECONDS = 30.0


def advisors(tmp_path: Path) -> Advisors:
    return Advisors(ACTIONS, "cpu", lambda state: tmp_path / state / "advisor.pt")


def test_inputs_hold_the_scaled_ram_and_a_fixed_number_of_fact_slots():
    inputs = encode(np.array([0, 255], dtype=np.uint8), {"lives": 3, "door": Offset(-4, 2), "target": "bell"})

    assert inputs.shape == (2 + FACT_SLOTS,)
    assert inputs[:2].tolist() == [0.0, 255 / RAM_SCALE]
    assert sorted(value for value in inputs[2:] if value) == [-4.0, 1.0, 2.0, 3.0]


def test_ppo_teaches_the_advisor_the_rewarded_move_and_saves_it(tmp_path: Path):
    pool = advisors(tmp_path)
    for _ in range(UPDATES):
        rollout = Rollout()
        for decision in pool.act(["Play"] * 64, [INPUTS] * 64):
            rollout.add(decision, float(decision.action), False, False)
        pool.learn("Play", rollout, [])
    pool.close()

    assert pool.advise("Play", INPUTS).action == 1
    assert pool.advise("Play", INPUTS).probabilities["right"] > 0.9
    reloaded = advisors(tmp_path)
    assert reloaded.advise("Play", INPUTS).probabilities["right"] == pool.advise("Play", INPUTS).probabilities["right"]


class EndlessPractice:
    def __init__(self) -> None:
        self.states = ["Play", "Play"]
        self.inputs = [INPUTS, INPUTS]
        self.steps = 0

    def send(self, actions: list[int]) -> None:
        self.actions = actions

    def receive(self) -> list[PracticeStep]:
        self.steps += 1
        return [PracticeStep(INPUTS, float(action), False, False, "Play", "Play") for action in self.actions]


def test_the_coach_practises_until_stopped_and_learns_every_full_rollout(tmp_path: Path):
    pool, practice = advisors(tmp_path), EndlessPractice()
    coach = Coach(pool, practice, LabRun(tmp_path / "no-lab-run"), lambda state: [])
    deadline = time.monotonic() + COACH_WAIT_SECONDS
    while practice.steps < ROLLOUT_STEPS and time.monotonic() < deadline:
        coach.check()
        time.sleep(0.01)
    coach.stop()
    pool.close()

    assert practice.steps >= ROLLOUT_STEPS // 2
    assert pool.models["Play"].num_timesteps >= ROLLOUT_STEPS
    assert torch.equal(pool.models["Play"].acting.policy[-1].weight, pool.models["Play"].learning.policy[-1].weight)


class ExitingPractice(EndlessPractice):
    def receive(self) -> list[PracticeStep]:
        self.steps += 1
        return [
            PracticeStep(INPUTS, 1.0, True, False, "Door", "Play"),
            PracticeStep(INPUTS, 0.0, False, False, "Play", "Play"),
        ]


def test_the_coach_counts_where_practice_leaves_each_state(tmp_path: Path):
    pool, practice = advisors(tmp_path), ExitingPractice()
    coach = Coach(pool, practice, LabRun(tmp_path / "no-lab-run"), lambda state: [])
    deadline = time.monotonic() + COACH_WAIT_SECONDS
    while practice.steps < 10 and time.monotonic() < deadline:
        coach.check()
        time.sleep(0.01)
    coach.stop()
    pool.close()

    assert coach.exits["Play"]["Door"] >= 10
    assert "Play" not in coach.exits["Play"]
