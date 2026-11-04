import random
import time
from dataclasses import replace
from pathlib import Path

import msgspec
import numpy as np
from fakes import saving_environment
from test_tas import NO_PREFERENCES, START, LineRunner

from datenwissenschaften.tas import archives as archives_module
from datenwissenschaften.tas import explore
from datenwissenschaften.tas.archives import KEPT_CELLS, Archive, Archives, fresh_archive, pruned
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
    runner = LineRunner()
    archive = fresh_archive(runner, START)
    return Exploration(runner, START, UNBOUNDED, time.monotonic() + 5.0, None, NO_PREFERENCES, archive)


def test_the_frontier_tries_every_short_input_from_a_place_and_finds_the_shortest_crossing():
    explored = exploration()

    frontier(explored, random.Random(1))

    assert explored.crossings[0].frames == 6
    assert len(explored.archive) > 1


def test_the_tree_search_finds_the_way_out_by_expanding_its_most_promising_moves():
    explored = exploration()

    tree(explored, random.Random(1))

    assert explored.crossings and explored.crossings[0].frames <= 8


def test_the_modes_take_turns_by_their_share_of_the_time(monkeypatch, tmp_path: Path):
    clock = [0.0]
    spent: dict[str, float] = {"quick": 0.0, "slow": 0.0}

    def mode(name: str, seconds: float):
        def run(explored: Exploration, rng: random.Random) -> None:
            clock[0] += seconds
            spent[name] += seconds

        return run

    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(explore, "STRATEGIES", (mode("quick", 0.001), mode("slow", 0.02)))
    monkeypatch.setattr(explore, "SHARES", (0.5, 0.5))

    explore.search_segment(LineRunner(), START, UNBOUNDED, 1, 0.4, None, NO_PREFERENCES, Archives(tmp_path, 2))

    assert abs(spent["quick"] - spent["slow"]) <= 0.02


def test_a_search_from_the_same_start_goes_on_from_the_places_the_last_one_found(tmp_path: Path):
    archives = Archives(tmp_path, 2)
    explore.search_segment(LineRunner(), START, UNBOUNDED, 1, 0.2, None, NO_PREFERENCES, archives)
    explored = archives.resumed(LineRunner(), START, UNBOUNDED)

    assert len(explored.cells) > 1
    assert len(archives.resumed(LineRunner(), START, UNBOUNDED - 1).cells) == 1


def test_the_places_a_search_found_survive_a_restart(tmp_path: Path):
    archives = Archives(tmp_path, 2)
    explore.search_segment(LineRunner(), START, UNBOUNDED, 1, 0.2, None, NO_PREFERENCES, archives)
    kept = archives.resumed(LineRunner(), START, UNBOUNDED)
    restarted = Archives(tmp_path, 2).resumed(LineRunner(), START, UNBOUNDED)

    assert {key: cell.frames for key, cell in restarted.cells.items()} == {
        key: cell.frames for key, cell in kept.cells.items()
    }
    assert (restarted.novelty.seen == kept.novelty.seen).all()
    assert all((cell.trail is None) == (restarted.cells[key].trail is None) for key, cell in kept.cells.items())


def test_only_the_newest_archives_stay_on_disk(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(archives_module, "KEPT_ARCHIVES", 2)
    archives = Archives(tmp_path, 2)
    for bound in range(3):
        archives.keep(START, bound, fresh_archive(LineRunner(), START))

    assert len([folder for folder in tmp_path.iterdir() if folder.is_dir()]) == 2


def test_each_search_process_keeps_its_own_file_and_a_restart_merges_them(tmp_path: Path, monkeypatch):
    explored = exploration()
    root = explored.archive[next(iter(explored.archive))]
    near, far = replace(root, frames=3), replace(root, frames=9)
    for pid, cells in ((1, {("spot",): far, ("only",): near}), (2, {("spot",): near})):
        monkeypatch.setattr(archives_module.os, "getpid", lambda pid=pid: pid)
        Archives(tmp_path, 2).keep(START, UNBOUNDED, Archive(cells, explored.novelty))

    restarted = Archives(tmp_path, 2).resumed(LineRunner(), START, UNBOUNDED)

    assert {key: cell.frames for key, cell in restarted.cells.items()} == {("spot",): 3, ("only",): 3}


def test_a_kept_archive_holds_the_best_scored_and_the_rarest_places():
    explored = exploration()
    root = explored.archive[next(iter(explored.archive))]
    scored = {index: replace(root, score=float(index), novelty=0) for index in range(KEPT_CELLS)}
    rare = {f"rare{index}": replace(root, score=-1.0, novelty=100) for index in range(5)}

    kept = pruned(Archive({**scored, **rare}, explored.novelty)).cells

    assert len(kept) == KEPT_CELLS
    assert rare.keys() <= kept.keys()
    assert KEPT_CELLS - 1 in kept


def test_a_map_saved_as_one_file_by_the_last_release_is_still_read_and_then_moved(tmp_path: Path):
    explored = exploration()
    root = explored.archive[next(iter(explored.archive))]
    single = tmp_path / f"{archives_module.archive_key(START, UNBOUNDED)}.msgpack"
    archives_module.save_archive(single, START.state, {("spot",): replace(root, frames=4)}, explored.novelty)
    archives = Archives(tmp_path, 2)

    resumed = archives.resumed(LineRunner(), START, UNBOUNDED)
    archives.keep(START, UNBOUNDED, resumed)

    assert {key: cell.frames for key, cell in resumed.cells.items()} == {("spot",): 4}
    assert not single.exists()
    assert Archives(tmp_path, 2).resumed(LineRunner(), START, UNBOUNDED).cells.keys() == {("spot",)}
