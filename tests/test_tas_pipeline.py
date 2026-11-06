import os
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
from fakes import saved_state, saving_environment, write_config

from datenwissenschaften.policy.prior import PolicyPrior
from datenwissenschaften.speedrun import feed as feed_module
from datenwissenschaften.speedrun import footage, movie, run, searcher, workers
from datenwissenschaften.speedrun.archives import Archives
from datenwissenschaften.speedrun.explore import search_segment
from datenwissenschaften.speedrun.hints import Hint, Press, hints_path, load_hints, pressed
from datenwissenschaften.speedrun.live import clear_reports
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.paths import TAS_MOVIE, atomic_write
from datenwissenschaften.speedrun.policy_start import (
    POLICY_EXITS_FILE,
    POLICY_START_FILE,
    load_policy_exits,
    load_policy_start,
    save_policy_exit,
)
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.prior import ActionPrior
from datenwissenschaften.speedrun.reel import Reel
from datenwissenschaften.speedrun.routes import (
    FULL_RUN,
    ROUTES_FILE,
    Leg,
    Part,
    Route,
    Routes,
    load_progress,
    save_progress,
)
from datenwissenschaften.speedrun.run import BACKTRACK_SEARCHES
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Checkpoint, Crossing, shave
from datenwissenschaften.speedrun.stuck import (
    FAILURES_FILE,
    STUCK_FILE,
    Stuck,
    bar,
    cleared,
    failed,
    load_barred,
    load_failures,
    load_stuck,
    unbar,
)
from datenwissenschaften.speedrun.turns import Turn, load_turn, save_turn

BOSS_FRAMES = 3
SECONDS = 0.05


class SyncPool:
    def submit(self, function, *arguments) -> Future:
        future: Future = Future()
        future.set_result(function(*arguments))
        return future

    def __enter__(self) -> "SyncPool":
        return self

    def __exit__(self, *exception: object) -> None:
        pass


def fake_render(paths: object, recordings: list[Path]) -> list[Path]:
    for recording in recordings:
        recording.with_suffix(".mp4").write_bytes(recording.read_bytes())
    return [recording.with_suffix(".mp4") for recording in recordings]


def fake_stack(left: Path, right: Path, stacked: Path) -> Path:
    stacked.write_bytes(left.read_bytes() + right.read_bytes())
    return stacked


def local_workers(monkeypatch, runner: SegmentRunner) -> None:
    monkeypatch.setattr(footage, "render_video", fake_render)
    monkeypatch.setattr(footage, "stack_videos", fake_stack)
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)
    monkeypatch.setattr(searcher, "explore", workers.explore)
    monkeypatch.setattr(searcher, "polish", workers.polish)


def local_tas(tmp_path: Path, monkeypatch, runner: SegmentRunner) -> run.Tas:
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    monkeypatch.setattr(run, "SegmentRunner", lambda wrapper: runner)
    local_workers(monkeypatch, runner)
    return run.tas_run("fakes:FakeWrapper", write_config(tmp_path), SyncPool())


def test_the_runner_restores_savestates_and_tells_when_a_segment_is_left(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    start = runner.power_on()

    outcomes = [runner.decide(0) for _ in range(BOSS_FRAMES)]
    runner.restore(start)

    assert [outcome.state for outcome in outcomes] == ["Survive", "Survive", "Boss"]
    assert runner.forward("Survive", "Boss") and not runner.forward("Boss", "Survive")
    assert runner.snapshot() == start.emulator_state == saved_state(0)
    assert runner.key(31) == ("Survive", 1)
    assert runner.remaining() is None


def test_without_a_position_the_cell_is_the_games_dynamics_not_the_time(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    runner.power_on()
    runner.ram = SimpleNamespace(position=lambda: None, dynamics=lambda: (3, 4))

    assert runner.key(31) == runner.key(900) == ("Survive", 3, 4)


def test_search_and_improve_write_a_verified_movie_and_keep_only_faster_segments(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    leg = Leg(FULL_RUN, runner.power_on(), "Boss")

    searched = tas.extend(leg, [], None)
    improved = tas.improve(leg, [Segment("Boss", [[1, 0]] * 6, bytes([6]), False)])
    run.publish(tas, improved)

    assert [(segment.state, len(segment.pressed)) for segment in searched] == [("Boss", BOSS_FRAMES)]
    assert run.frames(improved) == BOSS_FRAMES
    assert run.frames(run.load_progress(tas.routes.progress)) == BOSS_FRAMES
    assert (tas.output / TAS_MOVIE).read_bytes() == b"movie"
    assert (tas.output / "screen.mp4").read_bytes() == b"movie"
    assert (tas.output / "compare.mp4").read_bytes() == b"moviemovie"
    assert (tmp_path / "cache" / "sprites" / "catalog.json").read_bytes() == b"{}"


def test_a_movie_that_no_longer_reaches_its_segments_is_cut_back(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    power_on = runner.power_on()

    kept = tas.verified(Leg(FULL_RUN, power_on, "Boss"), [Segment("Boss", [[1, 0]], bytes([1]), False)])

    assert kept == []
    assert movie.verified_segments(runner, power_on, [], tas.output / "again.bk2", None).verified == 0
    assert tas.searcher.best(power_on, 1, 1) is None


def test_verify_replays_every_part_from_its_own_start_and_rejects_one_that_breaks(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    monkeypatch.setattr(run, "make_environment", lambda wrapper_cls, settings, worker: runner.wrapper)
    config = write_config(tmp_path)
    entry = runner.power_on().emulator_state

    def routes(name: str, *segments: tuple[str, int, bool]) -> Path:
        path = tmp_path / name / ROUTES_FILE
        route = Route(
            entry,
            [Segment(state, [[1, 0]] * frames, bytes([frames]), won) for state, frames, won in segments],
        )
        atomic_write(path, msgspec.msgpack.encode({"Survive": route}))
        return path

    run.verify("fakes:FakeWrapper", config, routes("good", ("Boss", BOSS_FRAMES, False)))
    run.verify("fakes:FakeWrapper", config, routes("newest_refused", ("Survive", 1, False), ("Boss", 1, False)))

    with pytest.raises(RuntimeError):
        run.verify("fakes:FakeWrapper", config, routes("older_broken", ("Boss", 1, False), ("Boss", 2, False)))
    with pytest.raises(RuntimeError):
        run.verify("fakes:FakeWrapper", config, routes("beaten_broken", ("Survive", 1, False), ("Boss", 1, True)))


def test_a_segment_without_a_way_out_is_marked_stuck_and_the_way_into_it_searched_again(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    stuck: list[Stuck | None] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | None:
        stuck.append(load_stuck(tas.output / STUCK_FILE))
        return found(self, start, bound, rounds) if len(stuck) > 1 else None

    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], None)

    assert run.frames(segments) == BOSS_FRAMES
    assert [marked and (marked.state, marked.searches) for marked in stuck] == [None, ("Survive", 1), None]
    assert load_stuck(tas.output / STUCK_FILE) is None
    assert [(event.kind, event.state) for event in tas.feed.journal.journal.events] == [
        ("stuck", "Survive"),
        ("found", "Boss"),
    ]


class SearchOver(Exception):
    pass


def test_a_state_the_search_finds_its_way_back_into_stays_stuck_since_its_first_failed_search(
    tmp_path: Path, monkeypatch
):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    stuck: list[Stuck | None] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | None:
        if start.state == "Survive":
            return found(self, start, bound, rounds)
        stuck.append(load_stuck(tas.output / STUCK_FILE))
        if len(stuck) > 2 * BACKTRACK_SEARCHES:
            raise SearchOver
        return None

    monkeypatch.setattr(searcher.Searcher, "best", best)

    with pytest.raises(SearchOver):
        tas.extend(Leg(FULL_RUN, runner.power_on(), "Exit"), [], None)

    first, *marked = stuck
    assert first is None
    assert [record and record.searches for record in marked] == list(range(1, 2 * BACKTRACK_SEARCHES + 1))
    assert {record and (record.state, record.since) for record in marked} == {("Boss", marked[0].since)}


def test_workers_search_and_shave_from_a_savestate(tmp_path: Path, monkeypatch):
    env = saving_environment(tmp_path)
    monkeypatch.setattr(workers, "make_environment", lambda wrapper_cls, settings, worker: env)
    runner = workers.worker_runner.__wrapped__("fakes:FakeWrapper", write_config(tmp_path))
    start = runner.power_on()
    preferences = Preferences(ActionPrior(tmp_path / "prior.msgpack"), PolicyPrior(tmp_path / "policy.npz"))
    archives = Archives(tmp_path / "archives", 2)
    crossing = search_segment(runner, start, 100, 1, SECONDS, None, preferences, archives).crossings[0]
    monkeypatch.setattr(workers, "worker_runner", lambda wrapper, config: runner)
    monkeypatch.setattr(workers, "worker_preferences", lambda config: preferences)
    monkeypatch.setattr(workers, "worker_archives", lambda config: archives)

    assert (
        workers.explore("fakes:FakeWrapper", tmp_path, start, 100, 1, SECONDS, None).crossings[0].frames == BOSS_FRAMES
    )
    assert workers.polish("fakes:FakeWrapper", tmp_path, start, crossing, SECONDS) == shave(runner, start, crossing, 0)


def test_every_new_piece_is_followed_by_one_try_to_shorten_an_earlier_one(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    shortened: list[int] = []
    monkeypatch.setattr(run.Tas, "shorten", lambda self, leg, segments, index: shortened.append(index) or segments)

    tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], None)

    assert shortened == [0]
    assert tas.turn == 1


def test_a_faster_piece_that_breaks_the_rest_of_the_run_is_not_taken(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    segments = [Segment("Survive", [[1, 0]] * 4, bytes([4]), False), Segment("Boss", [[1, 0]], bytes([5]), False)]
    faster = Crossing(Start("Survive", bytes([2])), np.zeros((2, 2), np.int8), False, ("Survive", 0))
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: faster)
    replayed = lambda runner, start, shorter, path, catalog: movie.Replay(1, {}, [bytes([2])])  # noqa: E731
    monkeypatch.setattr(run, "verified_segments", replayed)

    assert tas.shorten(Leg(FULL_RUN, runner.power_on(), "Boss"), segments, 0) is segments


def test_a_checkpoint_is_kept_as_a_piece_and_the_search_goes_on_from_it(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    checkpoint = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 1.0, None)
    calls: list[int] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        calls.append(bound)
        return checkpoint if len(calls) == 1 else found(self, start, bound, rounds)

    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], None)

    assert [(segment.state, len(segment.pressed)) for segment in segments] == [("Survive", 1), ("Boss", 2)]
    assert [event.kind for event in tas.feed.journal.journal.events][:2] == ["checkpoint", "found"]


def test_the_search_goes_back_a_piece_only_after_three_failed_searches(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    piece = [Segment("Survive", [[0, 0]], bytes([1]), False)]

    leg = Leg(FULL_RUN, runner.power_on(), "Boss")

    kept = [tas.stuck(leg, piece, Start("Survive", bytes([1])), None) for _ in range(BACKTRACK_SEARCHES)]

    assert kept == [piece, piece, []]
    assert tas.barred == {FULL_RUN: "Survive"}


def test_going_back_drops_every_checkpoint_in_a_row_down_to_the_last_piece(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    piece = Segment("Boss", [[0, 0]], bytes([1]), False)
    checkpoints = [Segment("Boss", [[0, 0]], bytes([index]), False) for index in (2, 3, 4)]
    leg = Leg(FULL_RUN, runner.power_on(), "Exit")

    kept = [tas.stuck(leg, [piece, *checkpoints], Start("Boss", bytes([4])), None) for _ in range(BACKTRACK_SEARCHES)]

    assert kept[-1] == [piece]
    assert tas.barred == {FULL_RUN: "Boss"}


def test_checkpoints_in_a_row_count_the_pieces_that_stay_in_one_state():
    leg = Leg(FULL_RUN, Start("Survive", b""), "Boss")
    survive = Segment("Survive", [[0, 0]], b"", False)

    assert run.checkpoints_in_a_row(leg, []) == 0
    assert run.checkpoints_in_a_row(leg, [survive, survive]) == 2
    assert run.checkpoints_in_a_row(leg, [survive, Segment("Boss", [[0, 0]], b"", False)]) == 0


def test_no_checkpoint_comes_back_in_a_state_the_search_just_left_one_behind(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    useless = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 20.0, None)
    calls: list[int] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        calls.append(bound)
        return useless if len(calls) == 1 else found(self, start, bound, rounds)

    monkeypatch.setattr(searcher.Searcher, "best", best)
    bar(tas.barred_file, FULL_RUN, "Survive")

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], None)

    assert [segment.state for segment in segments] == ["Boss"]
    assert tas.barred == {}


def test_a_level_keeps_its_barred_state_when_another_level_goes_back_later(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    useless = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 20.0, None)
    calls: list[int] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        calls.append(bound)
        return useless if len(calls) == 1 else found(self, start, bound, rounds)

    first = Leg("Level 1", runner.power_on(), "Boss")
    second = Leg("Level 2", Start("Boss", bytes([3])), "Exit")
    for _ in range(BACKTRACK_SEARCHES):
        tas.stuck(first, [Segment("Survive", [[0, 0]], bytes([1]), False)], Start("Survive", bytes([1])), None)
    for _ in range(BACKTRACK_SEARCHES):
        tas.stuck(second, [Segment("Boss", [[0, 0]], bytes([4]), False)], Start("Boss", bytes([4])), None)
    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(first, [], None)

    assert [segment.state for segment in segments] == ["Boss"]
    assert tas.barred == {"Level 2": "Boss"}


def test_a_barred_state_stays_barred_after_the_search_restarts(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    leg = Leg("Level 1", runner.power_on(), "Boss")
    for _ in range(BACKTRACK_SEARCHES):
        local_tas(tmp_path, monkeypatch, runner).stuck(
            leg, [Segment("Survive", [[0, 0]], bytes([1]), False)], Start("Survive", bytes([1])), None
        )

    assert local_tas(tmp_path, monkeypatch, runner).barred == {"Level 1": "Survive"}


def test_a_level_with_patience_rests_after_that_many_searches(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    searched: list[str] = []
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: searched.append(start.state))

    assert tas.extend(Leg(FULL_RUN, runner.power_on(), "Exit"), [], 2) == []
    assert searched == ["Survive", "Survive"]


@pytest.mark.parametrize(
    ("taken", "searches", "ended"),
    [
        (Turn(FULL_RUN, 2), 1, Turn(FULL_RUN, 3)),
        (Turn(FULL_RUN, 3), 3, Turn(FULL_RUN, 3)),
        (Turn("Other", 2), 3, Turn(FULL_RUN, 3)),
        (Turn(FULL_RUN, 5, 4), 2, Turn(FULL_RUN, 7, 4)),
        (Turn(FULL_RUN, 11, 10), 1, Turn(FULL_RUN, 12, 10)),
    ],
)
def test_a_turn_cut_by_a_restart_goes_on_with_the_searches_it_has_left(
    tmp_path: Path, monkeypatch, taken: Turn, searches: int, ended: Turn
):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    save_turn(tas.turn_file, taken)
    searched: list[str] = []
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: searched.append(start.state))

    tas.extend(Leg(FULL_RUN, runner.power_on(), "Exit"), [], 3)

    assert len(searched) == searches
    assert load_turn(tas.turn_file) == ended


def test_a_search_cut_by_a_restart_is_not_counted_so_the_level_searches_it_again(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    save_turn(tas.turn_file, Turn(FULL_RUN, 2))

    def restarted(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(searcher.Searcher, "best", restarted)

    with pytest.raises(KeyboardInterrupt):
        tas.extend(Leg(FULL_RUN, runner.power_on(), "Exit"), [], 3)

    assert load_turn(tas.turn_file) == Turn(FULL_RUN, 2)


def test_a_level_keeps_its_turn_while_its_searches_find_pieces(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    found = searcher.Searcher.best
    searched: list[str] = []

    def best(self: searcher.Searcher, start: Start, bound: int, rounds: int) -> Crossing | Checkpoint | None:
        if bound == UNBOUNDED:
            searched.append(start.state)
        return found(self, start, bound, rounds)

    monkeypatch.setattr(searcher.Searcher, "best", best)

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Exit"), [], 1)

    assert [segment.state for segment in segments] == ["Boss"]
    assert searched == ["Survive", "Boss"]


def test_a_checkpoint_chain_without_a_way_out_is_dropped_after_one_turn_and_its_state_barred(
    tmp_path: Path, monkeypatch
):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    checkpoint = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 1.0, None)
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: checkpoint)
    leg = Leg("Level 1", runner.power_on(), "Exit")
    patience = 1 + BACKTRACK_SEARCHES

    first = tas.extend(leg, [], patience)
    later = tas.extend(leg, tas.routes.segments(leg.name), patience)

    assert first == [] and later == [] and tas.routes.segments(leg.name) == []
    assert [event.kind for event in tas.feed.journal.journal.events] == ["checkpoint"] * (patience - 1) + ["stuck"] * (
        patience + 1
    )
    assert tas.barred == {"Level 1": "Survive"}


def test_checkpoints_on_the_tas_route_go_on_after_the_first_turn(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    guided = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 1.0, 300)
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: guided)
    leg = Leg("Level 1", runner.power_on(), "Exit")
    patience = 1 + BACKTRACK_SEARCHES

    tas.extend(leg, [], patience)

    assert [event.kind for event in tas.feed.journal.journal.events] == ["checkpoint"] * patience
    assert tas.routes.segments(leg.name) and all(piece.state == "Survive" for piece in tas.routes.segments(leg.name))
    assert tas.barred == {}


def test_a_route_checkpoint_no_closer_than_its_start_stops_after_the_first_turn(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    level = Checkpoint(Start("Survive", bytes([1])), np.zeros((1, 2), np.int8), 1.0, 300)
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: level)
    monkeypatch.setattr(SegmentRunner, "remaining", lambda self: 300)
    leg = Leg("Level 1", runner.power_on(), "Exit")
    patience = 1 + BACKTRACK_SEARCHES

    tas.extend(leg, [], patience)

    assert [event.kind for event in tas.feed.journal.journal.events] == ["checkpoint"] * (patience - 1) + ["stuck"]


def test_a_barred_state_stays_barred_when_the_search_finds_its_way_back_into_it(tmp_path: Path):
    barred = tmp_path / "barred.json"
    bar(barred, "Level 2", "Ledge")

    unbar(barred, "Level 2", "Level2")
    assert load_barred(barred) == {"Level 2": "Ledge"}
    unbar(barred, "Level 2", "Ledge")
    assert load_barred(barred) == {}


def test_worker_processes_compute_with_one_math_thread_each(monkeypatch):
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    run.worker_pool(1).shutdown()

    assert all(os.environ[name] == "1" for name in run.MATH_THREADS)


def test_every_level_with_pieces_gets_its_best_video_once_in_level_order(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    published: list[dict] = []
    monkeypatch.setattr(feed_module, "publish_metadata", lambda section, values: published.append(values))
    rendered: list[Path] = []
    monkeypatch.setattr(
        footage, "render_video", lambda paths, recordings: rendered.extend(recordings) or fake_render(paths, recordings)
    )
    parts = [Part("Survive", "Survive", "Boss"), Part("Level 1", "Boss", None)]
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", parts)
    routes.save(Leg("Level 1", Start("Boss", bytes([3])), None), [Segment("Boss", [[0, 0]] * 2, bytes([5]), False)])
    tas.feed.show([Segment("Boss", [[0, 0]], b"", False)], "Boss")
    shown = Reel(tas.settings, tas.feed, runner, routes)

    shown.show()
    shown.screens["Level 1"][1].settle()
    shown.show()
    shown.screens["Level 1"][1].settle()

    reel = published[-1]["reel"]
    assert [(level["part"], level["video"], level["frames"]) for level in reel] == [("Level 1", "levels/1.mp4", 2)]
    assert len(rendered) == 1


def test_pieces_in_states_the_game_package_dropped_keep_their_place_under_the_state_they_reach_now(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [Part("Survive", "Survive", "Boss")])
    gone = [Segment("Gone", [[0, 0]] * 1, bytes([1]), False), Segment("Boss", [[0, 0]] * 2, bytes([3]), False)]
    routes.save(Leg("Survive", Start("Survive", bytes([0])), "Boss"), gone)
    save_progress(tmp_path / "progress.msgpack", gone)

    assert routes.relabel(runner) == ["Survive", FULL_RUN]

    kept = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", routes.parts)
    assert [segment.state for segment in kept.segments("Survive")] == ["Survive", "Boss"]
    assert [segment.pressed for segment in kept.segments("Survive")] == [segment.pressed for segment in gone]
    assert [segment.state for segment in load_progress(tmp_path / "progress.msgpack")] == ["Survive", "Boss"]
    assert kept.relabel(runner) == []


def test_a_level_without_a_route_leaves_the_reel(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    routes = Routes(
        tmp_path / "levels.msgpack",
        tmp_path / "progress.msgpack",
        [Part("Survive", "Survive", "Boss"), Part("Level 1", "Boss", None)],
    )
    tas.feed.reeled("Level 1", {"number": 1, "part": "Level 1"})

    Reel(tas.settings, tas.feed, runner, routes).show()

    assert tas.feed.reel == {}


def test_a_practised_level_keeps_its_pieces_until_three_turns_failed_and_a_find_resets_the_count(
    tmp_path: Path, monkeypatch
):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    leg = Leg("Level 1", Start("Boss", bytes([3])), "Exit")
    piece = [Segment("Survive", [[0, 0]], bytes([1]), False)]
    patience = 3

    kept = [tas.stuck(leg, piece, Start("Survive", bytes([1])), patience) for _ in range(3 * patience)]

    assert [len(segments) for segments in kept] == [1] * (3 * patience - 1) + [0]
    cleared(tas.output / FAILURES_FILE, "Survive")
    assert load_failures(tas.output / FAILURES_FILE) == {}


def test_a_practised_state_stays_stuck_since_its_first_failed_search_after_other_levels_had_their_turn(
    tmp_path: Path, monkeypatch
):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    first = Leg("Level 1", runner.power_on(), "Boss")
    second = Leg("Level 2", Start("Boss", bytes([3])), "Exit")
    patience = 3

    tas.stuck(first, [], Start("Survive", bytes([0])), patience)
    since = load_stuck(tas.stuck_file).since
    tas.stuck(second, [], Start("Boss", bytes([3])), patience)
    tas.stuck(first, [], Start("Survive", bytes([0])), patience)

    assert load_stuck(tas.stuck_file) == Stuck("Survive", since, 2)


def test_a_practised_level_shows_its_own_stuck_state_from_the_start_of_its_turn(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    first = Leg("Level 1", runner.power_on(), "Boss")
    second = Leg("Level 2", Start("Boss", bytes([3])), "Exit")
    patience = 3
    tas.stuck(first, [], first.entry, patience)
    since = load_stuck(tas.stuck_file).since
    tas.stuck(second, [], second.entry, patience)
    shown: list[Stuck | None] = []
    monkeypatch.setattr(
        searcher.Searcher, "best", lambda self, start, bound, rounds: shown.append(load_stuck(tas.stuck_file))
    )

    tas.extend(first, [], patience)
    cleared(tas.output / FAILURES_FILE, first.entry.state)
    tas.extend(Leg("Level 1 again", runner.power_on(), "Boss"), [], 1)

    assert shown[0] == Stuck(first.entry.state, since, 1)
    assert shown[-1] is None


def test_failure_counts_saved_before_they_kept_their_first_failure_still_count(tmp_path: Path):
    path = tmp_path / FAILURES_FILE
    path.write_text('{"Level11": 18}')

    assert failed(path, "Level11").searches == 19
    assert load_failures(path)["Level11"].searches == 19


def test_the_lab_hint_leads_out_of_a_state_before_any_search(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    tas.hints = [Hint("Survive", [Press(BOSS_FRAMES, ["A"])])]
    bounds: list[int] = []
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: bounds.append(bound))

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], 1)

    assert [segment.state for segment in segments] == ["Boss"]
    assert UNBOUNDED not in bounds


def test_hints_are_read_from_next_to_the_config_and_name_real_buttons(tmp_path: Path):
    runner = SegmentRunner(saving_environment(tmp_path))
    config = write_config(tmp_path)
    hints_path(config).write_text('[{"state": "Survive", "presses": [[2, ["A"]], [1, []]]}]')

    hints = load_hints(hints_path(config))

    assert pressed(runner, hints[0]).tolist() == [[1, 0], [1, 0], [0, 0]]
    with pytest.raises(ValueError, match="presses B"):
        pressed(runner, Hint("Survive", [Press(1, ["B"])]))
    hints_path(config).unlink()
    with pytest.raises(RuntimeError, match="Missing"):
        load_hints(hints_path(config))


def test_a_way_out_the_policy_found_is_replayed_before_any_search(tmp_path: Path, monkeypatch):
    runner = SegmentRunner(saving_environment(tmp_path))
    tas = local_tas(tmp_path, monkeypatch, runner)
    save_policy_exit(tas.output / POLICY_EXITS_FILE, "Survive", np.ones((BOSS_FRAMES, 2), np.int8))
    bounds: list[int] = []
    monkeypatch.setattr(searcher.Searcher, "best", lambda self, start, bound, rounds: bounds.append(bound))

    segments = tas.extend(Leg(FULL_RUN, runner.power_on(), "Boss"), [], 1)

    assert [segment.state for segment in segments] == ["Boss"]
    assert UNBOUNDED not in bounds
    assert load_policy_start(tas.output / POLICY_START_FILE).state == "Survive"


def test_only_the_shortest_way_out_of_a_state_is_kept_for_the_search(tmp_path: Path):
    path = tmp_path / POLICY_EXITS_FILE

    assert save_policy_exit(path, "Boss", np.ones((4, 2), np.int8))
    assert not save_policy_exit(path, "Boss", np.ones((5, 2), np.int8))
    assert save_policy_exit(path, "Boss", np.ones((2, 2), np.int8))
    assert load_policy_exits(path) == {"Boss": [[1, 1], [1, 1]]}


def test_clearing_the_live_view_leaves_files_still_being_written(tmp_path: Path):
    (tmp_path / "1.jpg").write_bytes(b"tile")
    (tmp_path / ".2.jpg.7.tmp").write_bytes(b"half")

    clear_reports(tmp_path)

    assert [path.name for path in tmp_path.iterdir()] == [".2.jpg.7.tmp"]
