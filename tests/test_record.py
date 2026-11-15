from pathlib import Path
from types import SimpleNamespace

import msgspec
from fakes import FAKE_VERSION, FakeWrapper, SavingEmulator, fake_routes

from datenwissenschaften.speedrun.movie import Segment, frames
from datenwissenschaften.speedrun.record import RECORD_FILE, Record, StoredRecord, retired_path, stored_record
from datenwissenschaften.speedrun.routes import FULL_RUN, Leg, Part, Routes
from datenwissenschaften.speedrun.run import Tas
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.screen import Cut

OLD_VERSION = "0.9.0"
WINNING_SCRIPT = [(3, 0), (3, 1), (3, 5), (3, 6), (3, 7), (3, 7), (3, 7), (3, 7)]
STALE = bytes([9])
INTO_BOSS = Segment("Boss", [[1, 0]] * 2, STALE, False)
FAST = [INTO_BOSS, Segment("Boss", [[1, 0]] * 2, STALE, True)]
SLOW = [INTO_BOSS, Segment("Boss", [[1, 0]] * 3, STALE, True)]
BROKEN = [Segment("Survive", [[1, 0]] * 3, STALE, False), Segment("Boss", [[1, 0]] * 2, STALE, True)]


def winning_runner(tmp_path: Path) -> SegmentRunner:
    return SegmentRunner(FakeWrapper(SavingEmulator(WINNING_SCRIPT), fake_routes(tmp_path)))


def record(tmp_path: Path) -> Record:
    return Record(tmp_path / RECORD_FILE, winning_runner(tmp_path), FAKE_VERSION)


def stored(tmp_path: Path, version: str, segments: list[Segment]) -> None:
    (tmp_path / RECORD_FILE).write_bytes(msgspec.msgpack.encode(StoredRecord(version, segments)))


def test_a_won_run_from_power_on_becomes_the_record_and_is_restored_and_replayed_after_a_restart(tmp_path: Path):
    record(tmp_path).offer(SLOW)

    restored = record(tmp_path).segments

    assert [(piece.state, len(piece.pressed), piece.won) for piece in restored] == [
        ("Boss", 2, False),
        ("Boss", 3, True),
    ]
    assert [piece.emulator_state for piece in restored] == [bytes([2]), bytes([5])]


def test_only_a_faster_won_run_that_replays_from_power_on_replaces_the_record(tmp_path: Path):
    best = record(tmp_path)
    best.offer(SLOW)

    for slower_or_unfinished_or_broken in (SLOW, [INTO_BOSS], BROKEN):
        best.offer(slower_or_unfinished_or_broken)
        assert frames(best.segments) == frames(SLOW)
    best.offer(FAST)

    assert frames(best.segments) == frames(record(tmp_path).segments) == frames(FAST)


def test_the_record_is_shown_until_a_faster_run_is_verified(tmp_path: Path):
    best = record(tmp_path)

    assert best.shown([INTO_BOSS]) == [INTO_BOSS]
    assert frames(best.shown(SLOW)) == frames(best.shown([INTO_BOSS])) == frames(SLOW)
    assert frames(best.shown(FAST)) == frames(FAST)


def test_a_record_of_another_game_version_is_kept_when_it_replays_identically(tmp_path: Path):
    stored(tmp_path, OLD_VERSION, FAST)

    assert frames(record(tmp_path).segments) == frames(FAST)

    kept = stored_record(tmp_path / RECORD_FILE)
    assert kept is not None and kept.version == FAKE_VERSION
    assert not retired_path(tmp_path / RECORD_FILE, OLD_VERSION).exists()


def test_a_record_that_no_longer_replays_stays_the_record_of_its_version_and_the_search_starts_fresh(tmp_path: Path):
    stored(tmp_path, OLD_VERSION, BROKEN)

    assert record(tmp_path).segments == []

    retired = stored_record(retired_path(tmp_path / RECORD_FILE, OLD_VERSION))
    assert retired is not None and retired.version == OLD_VERSION and retired.segments == BROKEN
    assert stored_record(tmp_path / RECORD_FILE) is None


def test_the_feed_reports_the_record_while_the_search_works_on_a_level(tmp_path: Path):
    best = record(tmp_path)
    best.offer(SLOW)
    part = Part("Survive", "Survive", None)
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", [part], FAKE_VERSION)
    shown: list[tuple[int, str]] = []
    queued: list[Cut] = []
    course = SimpleNamespace(first="Survive", frame_rate=60.0)
    feed = SimpleNamespace(show=lambda segments, searching: shown.append((frames(segments), searching)), course=course)
    tas = SimpleNamespace(feed=feed, record=best, routes=routes, run=SimpleNamespace(queue=queued.append))

    Tas.show(tas, Leg("Survive", Start("Survive", bytes([0])), None), [INTO_BOSS], "Boss")
    Tas.show(tas, Leg(FULL_RUN, Start("Survive", bytes([0])), None), FAST, "Boss")

    assert shown == [(frames(SLOW), "Boss"), (frames(FAST), "Boss")]
    assert [(cut.field["part"], cut.field["frames"]) for cut in queued] == [
        (FULL_RUN, frames(SLOW)),
        (FULL_RUN, frames(FAST)),
    ]
