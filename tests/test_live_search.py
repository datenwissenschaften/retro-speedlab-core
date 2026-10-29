from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import fake_environment, write_config

from datenwissenschaften.route.position import Position
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas import feed as feed_module
from datenwissenschaften.tas.feed import SpeedrunFeed
from datenwissenschaften.tas.journal import JournalStore, Split, clock
from datenwissenschaften.tas.live import ATTEMPT_FILE, Watch, WorkerLive, WorkerReport, clear_reports, read_reports
from datenwissenschaften.tas.live_search import SearchRound, live_fields
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.report import Course, course_of, report, splits, start_frame
from datenwissenschaften.tas.runner import SegmentRunner, Start, key_spot
from datenwissenschaften.tas.search import Cell, Crossing

STARTED = datetime(2026, 10, 29, 14, 7, 10, tzinfo=UTC)
START = Start("Survive", b"")
LEVELS = {"Level 1": ("Menu", "Play"), "Level 2": ("Level2",)}


def course(tmp_path: Path) -> Course:
    game = load_config(write_config(tmp_path)).speedrun.game
    return Course(LEVELS, "Menu", 60.0, {"Play": "Eat a pibbley"}, 1, game)


def worker(seed: int, shows: bool, visits: list[tuple[int, int, int, int]], best: int | None) -> WorkerReport:
    return WorkerReport(seed, shows, 10, 600, best, visits, {"lives": 3}, Position(2, 1012, 999, 84, False))


def test_a_worker_writes_its_report_and_frame_and_the_main_process_reads_them_back(tmp_path: Path):
    runner = SegmentRunner(fake_environment(tmp_path, [(3, 4)]))
    archive = {
        ("Survive", 2, 8, 6, 0, False, ()): Cell(START, None, 0, 2, None, 0.0),
        ("Survive", 2, 9, 7, 1, True, ()): Cell(START, None, 0, 0, None, 0.0),
        ("Survive", 1): Cell(START, None, 0, 5, None, 0.0),
    }
    crossings = [Crossing(START, runner.wrapper.action_table[0], False, ("Boss",))]
    shown, quiet = WorkerLive(Watch(tmp_path, True), 7), WorkerLive(Watch(tmp_path, False), 3)

    shown.step(runner, archive, crossings, 4)
    shown.step(runner, archive, crossings, 4)
    quiet.step(runner, {}, [], 2)

    assert read_reports(tmp_path) == [
        WorkerReport(3, False, 0, 2, None, [], {"lives": 3, "score": 4}, None),
        WorkerReport(7, True, 3, 4, 1, [(2, 4, 3, 4)], {"lives": 3, "score": 4}, None),
    ]
    assert (tmp_path / ATTEMPT_FILE).read_bytes()[:2] == b"\xff\xd8"
    clear_reports(tmp_path)
    assert read_reports(tmp_path) == []
    assert key_spot(("Survive", -1, -3, 5, 0, False, ())) == (-1, -2, 2)


def test_a_report_that_vanishes_while_it_is_read_is_skipped(tmp_path: Path, monkeypatch):
    (tmp_path / "1.msgpack").write_bytes(b"")
    monkeypatch.setattr(Path, "read_bytes", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))

    assert read_reports(tmp_path) == []


def test_the_live_search_adds_up_its_workers_and_maps_the_start_area():
    search = SearchRound("Scale2", 0, 2, STARTED, 120.0, Position(2, 1012, 999, 84, False))
    reports = [worker(5, False, [(2, 63, 61, 4), (3, 1, 1, 9)], 393), worker(4, True, [(2, 63, 61, 8)], None)]
    route = [(1, 0, 0), (2, 63, 62), (2, 64, 62)]

    fields = live_fields(search, reports, route, STARTED + timedelta(seconds=2))

    assert fields["search"] == {
        "state": "Scale2",
        "round": 1,
        "rounds": 2,
        "started_at": "2026-10-29T14:07:10+00:00",
        "ends_at": "2026-10-29T14:09:10+00:00",
        "workers": [
            {"seed": 5, "cells": 10, "frames": 600, "best_frames": 393},
            {"seed": 4, "cells": 10, "frames": 600, "best_frames": None},
        ],
        "cells": 20,
        "frames": 1200,
        "frames_per_second": 600.0,
        "best_frames": 393,
    }
    assert fields["map"] == {
        "state": "Scale2",
        "area": 2,
        "cell_pixels": 16,
        "cells": [[63, 61, 12]],
        "route": [[63, 62], [64, 62]],
        "player": [63, 62],
    }
    assert fields["ram"] == {"lives": 3}
    assert fields["position"] == {"area": 2, "x": 1012, "y": 999, "height": 84}


def test_a_search_without_a_start_position_maps_its_busiest_area_and_none_without_cells():
    search = SearchRound("Menu", 1, 2, STARTED, 240.0, None)
    quiet = WorkerReport(1, True, 0, 0, None, [(3, 1, 1, 2)], {}, None)

    mapped = live_fields(search, [quiet], [], STARTED)
    empty = live_fields(search, [], [], STARTED)

    assert mapped["map"]["area"] == 3 and mapped["map"]["player"] is None
    assert mapped["position"] is None and mapped["search"]["frames_per_second"] == 0.0
    assert (empty["map"], empty["ram"]) == (None, None)
    assert live_fields(None, [quiet], [], STARTED) == {"search": None, "map": None, "ram": None, "position": None}


def test_the_journal_keeps_the_first_time_of_each_level_and_tells_when_a_level_got_faster(tmp_path: Path):
    path = tmp_path / "journal.json"
    journal = JournalStore(path)

    first = journal.track([Split("Level 1", 64.69, 64.69)], 64.69)
    journal.track([Split("Level 1", 64.69, 64.69)], 64.69)
    faster = journal.track([Split("Level 1", 62.2, 62.2)], 70.0)
    for index in range(25):
        journal.event("found", "Play", index)

    again = JournalStore(path).fields()
    assert first == faster == {"Level 1": 64.69}
    assert [milestone["text"] for milestone in again["milestones"]] == [
        "Level 1 beaten in 1:04.7",
        "Level 1 in 1:02.2, 2.5 s faster",
    ]
    assert [point["seconds"] for point in again["history"]] == [64.69, 70.0]
    assert [event["frames"] for event in again["events"]] == list(range(5, 25))
    assert clock(59.97) == "1:00.0"


def test_the_report_names_the_game_its_levels_goals_objectives_and_both_timings(tmp_path: Path):
    played = course(tmp_path)
    segments = [
        Segment("Play", [[0, 0]] * 30 + [[0, 1]] * 30, b"", False),
        Segment("Level2", [[0, 0]] * 120, b"", False),
    ]
    found = splits(played, segments)

    speedrun = report(played, segments, found, {"Level 1": 4.0}, "Play", None)

    assert speedrun["splits"] == [{"level": "Level 1", "seconds": 3.0, "level_seconds": 3.0, "first_seconds": 4.0}]
    assert speedrun["levels"] == [
        {"name": "Level 1", "states": ["Menu", "Play"]},
        {"name": "Level 2", "states": ["Level2"]},
    ]
    assert speedrun["goals"] == {"Play": "Eat a pibbley"}
    assert speedrun["objectives"] == [
        {"level": "game", "text": "Beat the game (Any%)"},
        {"level": "level", "text": "Beat Level 1"},
        {"level": "state", "text": "Eat a pibbley"},
    ]
    assert speedrun["game"] == {
        "id": "fakegame-v0",
        "title": "Fake Game",
        "platform": "NES",
        "category": "Any%",
        "sources": {"speedrun_com": {"game": "fake", "category": "any", "variables": {"level": "one"}}, "tasvideos": 7},
    }
    assert speedrun["timing"] == {"power_on_seconds": 3.0, "start_seconds": 2.5, "start_frame": 30}
    assert report(played, [], [], {}, "Menu", None)["timing"]["start_seconds"] is None
    assert start_frame(segments[1:], 1) is None


def test_the_course_reads_goals_and_the_start_button_and_fails_without_one(tmp_path: Path):
    runner = SegmentRunner(fake_environment(tmp_path, [(3, 0)]))
    game = load_config(write_config(tmp_path)).speedrun.game

    played = course_of(runner, game)
    runner.emulator.buttons = ["A", "B"]

    assert (played.first, played.start_button, played.frame_rate) == ("Survive", 1, 60.0)
    assert played.goals == {"Survive": "Survive", "Boss": "Beat the boss"}
    with pytest.raises(RuntimeError, match="START"):
        course_of(runner, game)


def test_the_feed_publishes_the_movie_with_the_live_search_and_notices_a_lab_run(tmp_path: Path, monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(feed_module, "publish_metadata", lambda section, values: published.append(values))
    marker = tmp_path / ".lab-run"
    journal = JournalStore(tmp_path / "journal.json")
    feed = SpeedrunFeed(course(tmp_path), journal, tmp_path / "live", tmp_path / "stuck.json", marker)
    segments = [Segment("Play", [[0, 1]] * 60, b"", False), Segment("Level2", [[0, 0]] * 60, b"", False)]

    feed.show(segments, "Level2")
    feed.follow([(2, 63, 62)])
    feed.begin(SearchRound("Level2", 0, 2, STARTED, 120.0, Position(2, 1012, 999, 84, False)))
    searching = published[-1]
    marker.write_text("{}", encoding="utf-8")
    feed.publish_every(SimpleNamespace(wait=lambda seconds: len(published) > 2))
    feed.end()

    assert searching["search"]["state"] == "Level2" and searching["map"]["route"] == [[63, 62]]
    assert searching["splits"][0]["first_seconds"] == 2.0 and searching["history"][0]["seconds"] == 2.0
    assert [(event["kind"], event["state"]) for event in published[-1]["events"]] == [("lab", "Level2")]
    assert published[-1]["search"] is None and "updated_at" in published[-1]
    assert feed.watch(0) == Watch(tmp_path / "live", True) and not feed.watch(1).shows
