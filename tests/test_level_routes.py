from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FAKE_VERSION

from datenwissenschaften.speedrun import levels
from datenwissenschaften.speedrun.levels import LevelTrainer
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import (
    FULL_RUN,
    ROUTES_FILE,
    Leg,
    Part,
    Routes,
    load_progress,
    parts_of,
)
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.stuck import STUCK_FILE, load_stuck, mark_stuck

COURSE = SimpleNamespace(first="Menu", levels={"Level 1": ("Play", "Heavy"), "Level 2": ("Level2", "Boss2")})
POWER_ON = Start("Menu", b"on")
NO_REEL = SimpleNamespace(show=lambda: None)
RUN = [
    Segment("Play", [[1]] * 3, b"play", False),
    Segment("Heavy", [[1]] * 4, b"heavy", False),
    Segment("Level2", [[1]] * 5, b"level2", False),
    Segment("Boss2", [[1]] * 6, b"boss2", False),
]


class OneRound(Exception):
    pass


def routes(tmp_path: Path) -> Routes:
    return Routes(tmp_path / ROUTES_FILE, tmp_path / "progress.msgpack", parts_of(COURSE), FAKE_VERSION)


def test_a_game_splits_into_its_opening_and_levels_each_ending_where_the_next_one_starts():
    assert parts_of(COURSE) == [
        Part("Menu", "Menu", "Play"),
        Part("Level 1", "Play", "Level2"),
        Part("Level 2", "Level2", None),
    ]
    assert parts_of(SimpleNamespace(first="Survive", levels={})) == [Part("Survive", "Survive", None)]


def test_the_old_run_becomes_one_route_per_part_with_a_fixed_start(tmp_path: Path):
    kept = routes(tmp_path)
    kept.split(RUN, POWER_ON)
    loaded = routes(tmp_path)

    assert [loaded.segments(part.name) for part in loaded.parts] == [RUN[:1], RUN[1:3], RUN[3:]]
    assert [loaded.entry(part, POWER_ON) for part in loaded.parts] == [
        POWER_ON,
        Start("Play", b"play"),
        Start("Level2", b"level2"),
    ]
    assert loaded.stitched() == RUN


def test_a_level_starts_only_once_the_one_before_it_is_beaten_and_shows_after_the_beaten_ones(tmp_path: Path):
    kept = routes(tmp_path)
    kept.split(RUN[:2], POWER_ON)
    menu, level_1, level_2 = kept.parts

    assert kept.entry(level_2, POWER_ON) is None
    assert kept.view(level_1.name, RUN[1:3]) == RUN[:3]
    assert kept.view(menu.name, []) == []
    assert kept.view(FULL_RUN, RUN[:1]) == RUN[:1]


def test_a_level_loaded_away_from_where_the_beaten_level_before_it_ends_is_not_shown_as_part_of_the_run(
    tmp_path: Path,
):
    kept = routes(tmp_path)
    kept.split(RUN[:3], POWER_ON)
    level_2 = kept.parts[2]
    kept.save(Leg(level_2.name, Start("Level2", b"cheat 1"), None), RUN[3:])

    assert kept.view(level_2.name, RUN[3:]) == RUN[:3]
    assert kept.stitched() == RUN[:3]


def test_the_full_run_is_its_own_movie(tmp_path: Path):
    kept = routes(tmp_path)
    kept.save(Leg(FULL_RUN, POWER_ON, None), RUN)

    assert load_progress(tmp_path / "progress.msgpack") == RUN
    assert not (tmp_path / ROUTES_FILE).exists()


def pieces(runner: object, kept: Routes) -> dict[str, int]:
    return {part.name: len(kept.segments(part.name)) for part in kept.parts if kept.segments(part.name)}


class FakePolisher:
    def __init__(self, calls: list) -> None:
        self.calls = calls

    def polish(self, part: Part, current: Part | None) -> bool:
        self.calls.append(("polish", part.name, current))
        return False


def trainer_tas(tmp_path: Path, calls: list, extend, improve) -> SimpleNamespace:
    feed = SimpleNamespace(practised=lambda frames: calls.append(("practised", frames)))
    turn_file = tmp_path / "turn.json"
    return SimpleNamespace(
        extend=extend,
        improve=improve,
        feed=feed,
        runner=None,
        settings=None,
        turn_file=turn_file,
        stuck_file=tmp_path / STUCK_FILE,
        output=tmp_path,
        record=SimpleNamespace(segments=[]),
    )


def test_the_trainer_beats_every_part_then_retrains_each_and_only_then_builds_the_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.save(Leg(FULL_RUN, POWER_ON, None), [*RUN[:3], Segment("Boss2", [[1]], b"won", True)])
    calls: list[tuple[str, str, int]] = []
    monkeypatch.setattr(levels, "Polisher", lambda tas, searches: FakePolisher(calls))

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        calls.append(("extend", leg.name, len(segments)))
        return segments

    def improve(leg: Leg, segments: list[Segment]) -> list[Segment]:
        calls.append(("improve", leg.name, len(segments)))
        if leg.name == FULL_RUN:
            raise OneRound
        return segments

    trainer = LevelTrainer(trainer_tas(tmp_path, calls, extend, improve), kept, POWER_ON, NO_REEL)
    with pytest.raises(OneRound):
        trainer.forever()

    practice = (
        "practised",
        {"Menu": 1, "Level 1": 2, "Level 2": 1},
    )
    assert calls == [
        practice,
        ("improve", "Menu", 1),
        ("polish", "Menu", None),
        ("improve", "Level 1", 2),
        ("polish", "Level 1", None),
        ("improve", "Level 2", 1),
        ("polish", "Level 2", None),
        practice,
        ("extend", FULL_RUN, 4),
        ("improve", FULL_RUN, 4),
    ]
    assert load_progress(tmp_path / "progress.msgpack") == []


def test_levels_are_beaten_one_after_the_other_each_from_where_the_level_before_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    calls: list = []
    beats = {"Level 1": [RUN[1], RUN[2]], "Level 2": [Segment("Boss2", [[1]], b"won", True)]}

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        calls.append((leg.name, leg.entry, patience))
        kept.save(leg, beats[leg.name])
        return segments

    def improve(leg: Leg, segments: list[Segment]) -> list[Segment]:
        raise OneRound

    trainer = LevelTrainer(trainer_tas(tmp_path, [], extend, improve), kept, POWER_ON, NO_REEL)
    with pytest.raises(OneRound):
        trainer.forever()

    assert calls == [
        ("Level 1", Start("Play", b"play"), None),
        ("Level 2", Start("Level2", b"level2"), None),
    ]


def test_a_level_is_searched_until_it_is_beaten_before_the_next_one_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    searched: list[str] = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        searched.append(leg.name)
        if len(searched) == 3:
            raise OneRound
        return segments

    trainer = LevelTrainer(trainer_tas(tmp_path, [], extend, None), kept, POWER_ON, NO_REEL)
    with pytest.raises(OneRound):
        trainer.beat_levels()

    assert searched == ["Level 1", "Level 1", "Level 1"]


def test_a_stuck_mark_left_from_before_every_level_was_beaten_is_cleared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.split([*RUN[:3], Segment("Boss2", [[1]], b"won", True)], POWER_ON)
    tas = trainer_tas(tmp_path, [], None, None)
    mark_stuck(tas.stuck_file, "Boss2")

    LevelTrainer(tas, kept, POWER_ON, NO_REEL).beat_levels()

    assert load_stuck(tas.stuck_file) is None


def test_a_level_shows_its_new_pieces_while_it_is_still_searched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    calls: list = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        kept.save(leg, RUN[1:2])
        rest()
        raise OneRound

    tas = trainer_tas(tmp_path, calls, extend, None)
    tas.routes = kept
    trainer = LevelTrainer(tas, kept, POWER_ON, NO_REEL)
    with pytest.raises(OneRound):
        trainer.beat_levels()

    assert calls == [("practised", {"Menu": 1}), ("practised", {"Menu": 1, "Level 1": 1})]


def test_without_levels_or_progress_the_record_splits_into_the_levels_and_starts_the_full_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    won = [*RUN[:3], Segment("Boss2", [[1]], b"won", True)]
    started: list[list[Segment]] = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        started.append(segments)
        raise OneRound

    tas = trainer_tas(tmp_path, [], extend, None)
    tas.record = SimpleNamespace(segments=won)
    trainer = LevelTrainer(tas, kept, POWER_ON, NO_REEL)
    with pytest.raises(OneRound):
        trainer.full_run()

    assert [kept.segments(part.name) for part in kept.parts] == [won[:1], won[1:3], won[3:]]
    assert started == [won]


def test_a_record_faster_than_the_levels_splits_into_the_levels_again(tmp_path: Path):
    kept = routes(tmp_path)
    slower = [*RUN[:3], Segment("Boss2", [[1]] * 9, b"slow won", True)]
    kept.split(slower, POWER_ON)
    won = [*RUN[:3], Segment("Boss2", [[1]], b"won", True)]
    tas = trainer_tas(tmp_path, [], None, None)
    tas.record = SimpleNamespace(segments=won)

    LevelTrainer(tas, kept, POWER_ON, NO_REEL)

    assert routes(tmp_path).stitched() == won


def test_levels_faster_than_the_record_stay(tmp_path: Path):
    kept = routes(tmp_path)
    won = [*RUN[:3], Segment("Boss2", [[1]], b"won", True)]
    kept.split(won, POWER_ON)
    tas = trainer_tas(tmp_path, [], None, None)
    tas.record = SimpleNamespace(segments=[*RUN[:3], Segment("Boss2", [[1]] * 9, b"slow won", True)])

    LevelTrainer(tas, kept, POWER_ON, NO_REEL)

    assert routes(tmp_path).stitched() == won


def full_run_start(tmp_path: Path, progress: list[Segment], record: list[Segment]) -> list[Segment]:
    kept = routes(tmp_path)
    kept.split(record, POWER_ON)
    kept.save(Leg(FULL_RUN, POWER_ON, None), progress)
    started: list[list[Segment]] = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None, rest: Callable[[], bool]) -> list[Segment]:
        started.append(segments)
        raise OneRound

    tas = trainer_tas(tmp_path, [], extend, None)
    tas.record = SimpleNamespace(segments=record)
    with pytest.raises(OneRound):
        LevelTrainer(tas, kept, POWER_ON, NO_REEL).full_run()
    return started[0]


def test_the_full_run_starts_from_the_record_when_the_stored_progress_is_slower(tmp_path: Path):
    won = [*RUN[:3], Segment("Boss2", [[1]], b"won", True)]
    slower = [*RUN[:3], Segment("Boss2", [[1]] * 9, b"slow won", True)]

    assert full_run_start(tmp_path, slower, won) == won


def test_the_full_run_continues_its_progress_when_it_is_faster_than_the_record(tmp_path: Path):
    won = [*RUN[:3], Segment("Boss2", [[1]], b"won", True)]
    slower = [*RUN[:3], Segment("Boss2", [[1]] * 9, b"slow won", True)]

    assert full_run_start(tmp_path, won, slower) == won
