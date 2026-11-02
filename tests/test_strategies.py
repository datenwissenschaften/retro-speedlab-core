import random
import time
from pathlib import Path

import msgspec
import numpy as np
from fakes import saving_environment
from test_tas import NO_PREFERENCES, START, LineRunner

from datenwissenschaften.tas import explore
from datenwissenschaften.tas.exploration import Exploration
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.novelty import RamNovelty
from datenwissenschaften.tas.prior import ActionPrior, learned_table
from datenwissenschaften.tas.routes import Leg, Part, Routes
from datenwissenschaften.tas.runner import SegmentRunner, Start
from datenwissenschaften.tas.search import UNBOUNDED
from datenwissenschaften.tas.strategies import frontier
from datenwissenschaften.tas.tree import tree


def test_ram_novelty_counts_the_byte_values_never_seen_before():
    novelty = RamNovelty(np.asarray([1, 2], dtype=np.uint8))

    assert novelty.see(np.asarray([1, 2], dtype=np.uint8)) == 0
    assert novelty.see(np.asarray([3, 2], dtype=np.uint8)) == 1
    assert novelty.see(np.asarray([3, 4], dtype=np.uint8)) == 1


def test_the_prior_prefers_what_worked_at_this_place_then_in_this_state(tmp_path: Path):
    path = tmp_path / "prior.msgpack"
    path.write_bytes(msgspec.msgpack.encode({"Walk|3": {2: 5}, "Walk": {1: 3, 7: 9}}))
    prior, rng = ActionPrior(path), random.Random(1)

    assert prior.choose(("Walk|3", "Walk"), 3, rng) == 2
    assert prior.choose(("Walk|9", "Walk"), 3, rng) == 1
    assert prior.choose(("Jump|1", "Jump"), 3, rng) is None
    assert ActionPrior(tmp_path / "none.msgpack").choose(("Walk|3", "Walk"), 3, rng) is None


def test_the_prior_learns_the_actions_of_the_stored_pieces(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", None)])
    routes.save(
        Leg("Survive", Start("Survive", bytes([0])), None), [Segment("Boss", [[1, 0], [0, 1], [1, 1]], b"", False)]
    )

    assert learned_table(runner, routes) == {"Survive": {0: 1, 1: 1}}


def exploration() -> Exploration:
    return Exploration(LineRunner(), START, UNBOUNDED, time.monotonic() + 5.0, None, NO_PREFERENCES)


def test_the_frontier_tries_every_short_input_from_a_place_and_finds_the_shortest_crossing():
    explored = exploration()

    frontier(explored, random.Random(1))

    assert explored.crossings[0].frames == 6
    assert len(explored.archive) > 1


def test_the_tree_search_finds_the_way_out_by_expanding_its_most_promising_moves():
    explored = exploration()

    tree(explored, random.Random(1))

    assert explored.crossings and explored.crossings[0].frames <= 8


def test_the_modes_take_turns_by_their_share_of_the_time(monkeypatch):
    spent: dict[str, float] = {"quick": 0.0, "slow": 0.0}

    def mode(name: str, seconds: float):
        def run(explored: Exploration, rng: random.Random) -> None:
            time.sleep(seconds)
            spent[name] += seconds

        return run

    monkeypatch.setattr(explore, "STRATEGIES", (mode("quick", 0.001), mode("slow", 0.02)))
    monkeypatch.setattr(explore, "SHARES", (0.5, 0.5))

    explore.search_segment(LineRunner(), START, UNBOUNDED, 1, 0.4, None, NO_PREFERENCES)

    assert abs(spent["quick"] - spent["slow"]) < 0.05
