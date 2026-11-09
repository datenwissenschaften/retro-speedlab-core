from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fakes import NO_EXPERIMENT, fake_catalog, fake_environment, write_config

from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun import feed as feed_module
from datenwissenschaften.speedrun.feed import SpeedrunFeed
from datenwissenschaften.speedrun.journal import JournalStore, Split, clock
from datenwissenschaften.speedrun.live import WorkerLive, WorkerReport, clear_reports, read_reports
from datenwissenschaften.speedrun.live_search import SearchRound, search_field
from datenwissenschaften.speedrun.mosaic import compose, shown_tiles, tile
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.report import Course, course_of, report, splits, start_frame
from datenwissenschaften.speedrun.routes import PartProgress
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Cell, Crossing

STARTED = datetime(2026, 10, 29, 14, 7, 10, tzinfo=UTC)
START = Start("Survive", b"")
LEVELS = {"Level 1": ("Menu", "Play"), "Level 2": ("Level2",)}


def course(tmp_path: Path) -> Course:
    game = load_config(write_config(tmp_path)).speedrun.game
    return Course(LEVELS, "Menu", 60.0, {"Play": "Eat a pibbley"}, 1, game)


def test_a_worker_writes_its_report_and_tile_and_the_main_process_reads_them_back(tmp_path: Path):
    runner = SegmentRunner(fake_environment(tmp_path, [(3, 4)]))
    archive = {
        ("Survive", 1): Cell(START, None, 0, 5, None, 0.0, 0),
        ("Survive", 2): Cell(START, None, 0, 0, None, 0.0, 0),
    }
    crossings = [Crossing(START, runner.wrapper.action_table[0], False, ("Boss",))]
    busy, quiet = WorkerLive(tmp_path, 7, "scout", "frontier"), WorkerLive(tmp_path, 3, "ppo", "policy")

    busy.step(runner, len(archive), crossings[0].frames, 4, 300)
    busy.step(runner, len(archive), crossings[0].frames, 4, 300)
    quiet.step(runner, 0, None, 2, None)

    expected = [WorkerReport(3, "ppo", "policy", 0, 2, None, None), WorkerReport(7, "scout", "frontier", 2, 4, 1, 300)]
    assert read_reports(tmp_path) == expected
    assert (tmp_path / "7.jpg").read_bytes()[:2] == b"\xff\xd8"
    clear_reports(tmp_path)
    assert read_reports(tmp_path) == [] and compose(tmp_path) is None


def test_a_report_that_vanishes_while_it_is_read_is_skipped(tmp_path: Path, monkeypatch):
    (tmp_path / "1.msgpack").write_bytes(b"")
    (tmp_path / "1.jpg").write_bytes(b"")
    monkeypatch.setattr(Path, "read_bytes", lambda path: (_ for _ in ()).throw(FileNotFoundError(path)))

    assert read_reports(tmp_path) == [] and shown_tiles(tmp_path) == []


def test_the_mosaic_shows_the_first_twelve_emulators_in_seed_order_in_three_rows_of_four(tmp_path: Path):
    for seed in range(13, 0, -1):
        (tmp_path / f"{seed}.jpg").write_bytes(tile(np.full((224, 256, 3), seed * 18, np.uint8)))

    mosaic = compose(tmp_path)
    picture = cv2.imdecode(np.frombuffer(mosaic.jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)

    assert mosaic.seeds == list(range(1, 13))
    assert picture.shape == (3 * 224, 4 * 256)
    assert abs(int(picture[112, 128]) - 18) < 4 and abs(int(picture[560, 896]) - 216) < 4


def test_the_mosaic_skips_files_that_are_not_a_worker_tile(tmp_path: Path):
    (tmp_path / "attempt.jpg").write_bytes(b"")
    (tmp_path / "4.jpg").write_bytes(tile(np.zeros((224, 256, 3), np.uint8)))

    assert compose(tmp_path).seeds == [4]


def test_two_workers_share_one_row_of_the_mosaic(tmp_path: Path):
    for seed in (5, 4):
        (tmp_path / f"{seed}.jpg").write_bytes(tile(np.zeros((224, 256, 3), np.uint8)))

    mosaic = compose(tmp_path)

    assert mosaic.seeds == [4, 5]
    assert cv2.imdecode(np.frombuffer(mosaic.jpeg, np.uint8), cv2.IMREAD_COLOR).shape == (224, 512, 3)


def test_the_live_search_adds_up_its_workers():
    search = SearchRound("Scale2", 0, 2, STARTED, 120.0, 393)
    reports = [
        WorkerReport(5, "scout", "walk", 10, 600, 393, 120),
        WorkerReport(4, "backtracker", "tree", 10, 600, None, 90),
    ]

    assert search_field(search, reports, STARTED + timedelta(seconds=2)) == {
        "state": "Scale2",
        "round": 1,
        "rounds": 2,
        "beat": 393,
        "started_at": "2026-10-29T14:07:10+00:00",
        "ends_at": "2026-10-29T14:09:10+00:00",
        "workers": [
            {
                "seed": 5,
                "role": "scout",
                "mode": "walk",
                "cells": 10,
                "frames": 600,
                "best_frames": 393,
                "closest": 120,
            },
            {
                "seed": 4,
                "role": "backtracker",
                "mode": "tree",
                "cells": 10,
                "frames": 600,
                "best_frames": None,
                "closest": 90,
            },
        ],
        "cells": 20,
        "frames": 1200,
        "frames_per_second": 600.0,
        "best_frames": 393,
        "closest": 90,
    }
    assert search_field(SearchRound("Menu", 1, 2, STARTED, 240.0, None), [], STARTED)["frames_per_second"] == 0.0
    assert search_field(None, reports, STARTED) is None


def test_the_journal_keeps_the_first_time_of_each_part_and_tells_only_of_levels_beaten_or_faster(tmp_path: Path):
    path = tmp_path / "journal.json"
    journal = JournalStore(path)

    opened = journal.track([Split("Menu", 10.8, 10.8)], 10.8, LEVELS)
    first = journal.track([Split("Level 1", 64.69, 64.69)], 64.69, LEVELS)
    journal.track([Split("Level 1", 64.69, 64.69)], 64.69, LEVELS)
    faster = journal.track([Split("Level 1", 62.2, 62.2)], 70.0, LEVELS)
    for index in range(25):
        journal.event("found", "Play", index)

    again = JournalStore(path).fields()
    assert opened == {"Menu": 10.8} and first == faster == {"Menu": 10.8, "Level 1": 64.69}
    assert [milestone["text"] for milestone in again["milestones"]] == [
        "Level 1 beaten in 1:04.7",
        "Level 1 in 1:02.2, 2.5 s faster",
    ]
    assert [point["seconds"] for point in again["history"]] == [10.8, 64.69, 70.0]
    assert [event["frames"] for event in again["events"]] == list(range(5, 25))
    assert clock(59.97) == "1:00.0"


def test_the_journal_tells_when_a_state_gets_closer_to_its_door(tmp_path: Path):
    path = tmp_path / "journal.json"
    journal = JournalStore(path)

    journal.closer("Level4Upper", 394)
    journal.closer("Level4Upper", 400)
    journal.closer("Level4Upper", 145)

    again = JournalStore(path).fields()
    assert again["closest"] == {"Level4Upper": 145}
    assert [milestone["text"] for milestone in again["milestones"]] == [
        "Level 4 Upper: 394 frames before the goal",
        "Level 4 Upper: 145 frames before the goal (was 394)",
    ]


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
        "sources": {"speedrun_com": {"game": "fake", "category": "any", "variables": {"level": "one"}}},
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
    feed = SpeedrunFeed(
        course(tmp_path),
        journal,
        tmp_path / "live",
        tmp_path / "stuck.json",
        marker,
        NO_EXPERIMENT,
        fake_catalog(tmp_path),
    )
    segments = [Segment("Play", [[0, 1]] * 60, b"", False), Segment("Level2", [[0, 0]] * 60, b"", False)]

    feed.show(segments, "Level2")
    feed.begin(SearchRound("Level2", 0, 2, STARTED, 120.0, None))
    searching = published[-1]
    marker.write_text("{}", encoding="utf-8")
    feed_module.publish_every(feed, SimpleNamespace(wait=lambda seconds: len(published) > 2), False)
    feed.end()
    feed.shown("movie", {"version": "abc", "frames": 120})
    feed.shown("compare", {"version": "def"})

    assert searching["movie"] is None and published[-1]["movie"] == {"version": "abc", "frames": 120}
    assert searching["search"]["state"] == "Level2"
    assert searching["splits"][0]["first_seconds"] == 2.0 and searching["history"][0]["seconds"] == 2.0
    assert [(event["kind"], event["state"]) for event in published[-2]["events"]] == [("lab", "Level2")]
    assert published[-2]["search"] is None and "updated_at" in published[-1]
    assert published[-1]["compare"] == {"version": "def"} and searching["compare"] is None
    assert published[-1]["sprites"] == [] and published[-1]["knowledge"]["player"] is None


def test_the_feed_publishes_nothing_before_it_has_a_run_to_show(tmp_path: Path, monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(feed_module, "publish_metadata", lambda section, values: published.append(values))
    journal = JournalStore(tmp_path / "journal.json")
    feed = SpeedrunFeed(
        course(tmp_path),
        journal,
        tmp_path / "live",
        tmp_path / "stuck.json",
        tmp_path / ".lab-run",
        NO_EXPERIMENT,
        fake_catalog(tmp_path),
    )

    feed.publish()
    feed.shown("movie", {"version": "abc", "frames": 120})

    assert published == []


def test_the_part_before_the_first_level_is_split_like_a_level(tmp_path: Path):
    game = load_config(write_config(tmp_path)).speedrun.game
    played = Course({"Level 1": ("Play", "Door"), "Level 2": ("Level2",)}, "Menu", 60.0, {}, 1, game)
    segments = [
        Segment("Play", [[0, 1]] * 60, b"", False),
        Segment("Door", [[0, 0]] * 30, b"", False),
        Segment("Level2", [[0, 0]] * 30, b"", False),
    ]

    assert splits(played, segments) == [Split("Menu", 1.0, 1.0), Split("Level 1", 2.0, 1.0)]
    assert report(played, segments, [], {}, "Level2", None)["opening"] == "Menu"


def test_checkpoint_pieces_never_end_a_level(tmp_path: Path):
    game = load_config(write_config(tmp_path)).speedrun.game
    played = Course({"Level 1": ("Play", "Door"), "Level 2": ("Level2",)}, "Menu", 60.0, {}, 1, game)
    segments = [
        Segment("Play", [[0, 1]] * 60, b"", False),
        Segment("Door", [[0, 0]] * 30, b"", False),
        Segment("Door", [[0, 0]] * 30, b"", False),
        Segment("Door", [[0, 0]] * 30, b"", False),
    ]

    assert [split.level for split in splits(played, segments)] == ["Menu"]


def test_the_feed_publishes_how_far_every_level_got_in_seconds(tmp_path: Path, monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(feed_module, "publish_metadata", lambda section, values: published.append(values))
    journal = JournalStore(tmp_path / "journal.json")
    feed = SpeedrunFeed(
        course(tmp_path),
        journal,
        tmp_path / "live",
        tmp_path / "stuck.json",
        tmp_path / ".lab-run",
        NO_EXPERIMENT,
        fake_catalog(tmp_path),
    )
    feed.show([Segment("Play", [[0, 1]] * 60, b"", False)], "Play")

    feed.practised({"Level 2": PartProgress("Level2", 3, 90, False, True, {"weight": 12})})

    assert published[-1]["practice"] == {
        "Level 2": {
            "reached": "Level2",
            "pieces": 3,
            "frames": 90,
            "beaten": False,
            "loaded": True,
            "details": {"weight": 12},
            "seconds": 1.5,
        }
    }


def test_the_feed_publishes_how_much_of_the_reference_video_each_level_matched(tmp_path: Path, monkeypatch):
    published: list[dict] = []
    monkeypatch.setattr(feed_module, "publish_metadata", lambda section, values: published.append(values))
    journal = JournalStore(tmp_path / "journal.json")
    feed = SpeedrunFeed(
        course(tmp_path),
        journal,
        tmp_path / "live",
        tmp_path / "stuck.json",
        tmp_path / ".lab-run",
        NO_EXPERIMENT,
        fake_catalog(tmp_path),
    )
    feed.show([Segment("Play", [[0, 1]] * 60, b"", False)], "Play")
    before = published[-1]["reference_progress"]

    feed.referenced([{"level": 0, "matched": 40, "frames": 300}, {"level": 7, "matched": 0, "frames": 90}])

    assert before == {}
    assert published[-1]["reference_progress"] == {
        next(iter(course(tmp_path).levels)): {"matched": 40, "frames": 300},
        "Level 8": {"matched": 0, "frames": 90},
    }
