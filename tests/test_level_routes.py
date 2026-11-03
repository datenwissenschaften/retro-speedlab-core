from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften.tas import levels
from datenwissenschaften.tas.levels import LevelTrainer
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.routes import (
    FULL_RUN,
    ROUTES_FILE,
    Leg,
    Part,
    Routes,
    load_progress,
    parts_of,
    save_progress,
)
from datenwissenschaften.tas.runner import Start
from datenwissenschaften.tas.turns import Turn, load_turn, save_turn

COURSE = SimpleNamespace(first="Menu", levels={"Level 1": ("Play", "Heavy"), "Level 2": ("Level2", "Boss2")})
POWER_ON = Start("Menu", b"on")
NO_REEL = SimpleNamespace(show=lambda: None)
LOADER = levels.LevelLoader(b"loader", 2)
RUN = [
    Segment("Play", [[1]] * 3, b"play", False),
    Segment("Heavy", [[1]] * 4, b"heavy", False),
    Segment("Level2", [[1]] * 5, b"level2", False),
    Segment("Boss2", [[1]] * 6, b"boss2", False),
]


class OneRound(Exception):
    pass


def routes(tmp_path: Path) -> Routes:
    return Routes(tmp_path / ROUTES_FILE, tmp_path / "progress.msgpack", parts_of(COURSE))


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


def trainer_tas(tmp_path: Path, calls: list, extend, improve) -> SimpleNamespace:
    feed = SimpleNamespace(practised=lambda frames: calls.append(("practised", frames)))
    turn_file = tmp_path / "turn.json"
    return SimpleNamespace(
        extend=extend, improve=improve, feed=feed, runner=None, settings=None, turn_file=turn_file, output=tmp_path
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

    def extend(leg: Leg, segments: list[Segment], patience: int | None) -> list[Segment]:
        calls.append(("extend", leg.name, len(segments)))
        return segments

    def improve(leg: Leg, segments: list[Segment]) -> list[Segment]:
        calls.append(("improve", leg.name, len(segments)))
        if leg.name == FULL_RUN:
            raise OneRound
        return segments

    trainer = LevelTrainer(trainer_tas(tmp_path, calls, extend, improve), kept, POWER_ON, None, NO_REEL)
    with pytest.raises(OneRound):
        trainer.forever()

    practice = (
        "practised",
        {"Menu": 1, "Level 1": 2, "Level 2": 1},
    )
    assert calls == [
        practice,
        ("improve", "Menu", 1),
        ("improve", "Level 1", 2),
        ("improve", "Level 2", 1),
        practice,
        ("extend", FULL_RUN, 4),
        ("improve", FULL_RUN, 4),
    ]
    assert load_progress(tmp_path / "progress.msgpack") == []


def test_unbeaten_levels_take_turns_and_a_level_without_its_part_before_starts_from_the_level_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    monkeypatch.setattr(
        levels, "level_start", lambda runner, loader, level, state: Start(state, f"cheat {level}".encode())
    )
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    calls: list = []
    beats = {"Level 1": [RUN[1], RUN[2]], "Level 2": [Segment("Boss2", [[1]], b"won", True)]}

    def extend(leg: Leg, segments: list[Segment], patience: int | None) -> list[Segment]:
        calls.append((leg.name, leg.entry, patience))
        if leg.name == "Level 2" or len(calls) > 2:
            kept.save(leg, beats[leg.name])
        return segments

    def improve(leg: Leg, segments: list[Segment]) -> list[Segment]:
        raise OneRound

    trainer = LevelTrainer(
        trainer_tas(tmp_path, [], extend, improve), kept, POWER_ON, levels.LevelLoader(b"loader", 2), NO_REEL
    )
    with pytest.raises(OneRound):
        trainer.forever()

    patience = levels.PRACTICE_SEARCHES
    assert calls == [
        ("Level 1", Start("Play", b"play"), patience),
        ("Level 2", Start("Level2", b"cheat 1"), patience),
        ("Level 1", Start("Play", b"play"), patience),
    ]


def test_an_unbeaten_level_loaded_before_the_level_before_it_was_beaten_starts_where_that_level_now_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    kept = routes(tmp_path)
    kept.split(RUN[:3], POWER_ON)
    kept.save(Leg("Level 2", Start("Level2", b"cheat 1"), None), [Segment("Level2", [[1]], b"dead end", False)])
    calls: list = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None) -> list[Segment]:
        calls.append((leg.name, leg.entry))
        raise OneRound

    trainer = LevelTrainer(
        trainer_tas(tmp_path, [], extend, None), kept, POWER_ON, levels.LevelLoader(b"loader", 2), NO_REEL
    )
    with pytest.raises(OneRound):
        trainer.forever()

    assert calls == [("Level 2", Start("Level2", b"level2"))]


@pytest.mark.parametrize(
    ("turn", "first"),
    [
        (Turn("Level 2", 1), "Level 2"),
        (Turn("Level 1", levels.PRACTICE_SEARCHES), "Level 2"),
        (Turn("Level 1", levels.PRACTICE_SEARCHES, 1), "Level 1"),
        (None, "Level 1"),
    ],
)
def test_after_a_restart_the_levels_take_turns_from_the_turn_that_was_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, turn: Turn | None, first: str
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    monkeypatch.setattr(levels, "level_start", lambda runner, loader, level, state: Start(state, b"cheat"))
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    if turn is not None:
        save_turn(tmp_path / "turn.json", turn)
    practised: list[str] = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None) -> list[Segment]:
        practised.append(leg.name)
        raise OneRound

    trainer = LevelTrainer(
        trainer_tas(tmp_path, [], extend, None), kept, POWER_ON, levels.LevelLoader(b"loader", 2), NO_REEL
    )
    with pytest.raises(OneRound):
        trainer.beat_levels()

    assert practised == [first]


def test_after_the_game_package_changes_the_states_of_a_level_that_level_takes_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(levels, "route_path", lambda settings: tmp_path / "route.json")
    monkeypatch.setattr(levels, "write_practice_route", lambda runner, kept, path, done: None)
    monkeypatch.setattr(levels, "write_prior", lambda runner, kept, path: None)
    monkeypatch.setattr(levels, "practised", pieces)
    monkeypatch.setattr(levels, "level_start", lambda runner, loader, level, state: Start(state, b"cheat"))
    kept = routes(tmp_path)
    kept.split(RUN[:1], POWER_ON)
    tas = trainer_tas(tmp_path, [], None, None)
    practised: list[str] = []

    def extend(leg: Leg, segments: list[Segment], patience: int | None) -> list[Segment]:
        practised.append(leg.name)
        raise OneRound

    tas.extend = extend
    tas.feed.course = COURSE
    levels.turn_to_changed_level(tas)
    save_turn(tas.turn_file, Turn("Level 1", 1))
    levels.turn_to_changed_level(tas)
    assert load_turn(tas.turn_file) == Turn("Level 1", 1)
    tas.feed.course = SimpleNamespace(first="Menu", levels={**COURSE.levels, "Level 2": ("Level2", "Bridge2", "Boss2")})
    levels.turn_to_changed_level(tas)

    trainer = LevelTrainer(tas, kept, POWER_ON, levels.LevelLoader(b"loader", 2), NO_REEL)
    with pytest.raises(OneRound):
        trainer.beat_levels()

    assert practised == ["Level 2"]


def test_pieces_in_states_the_game_package_dropped_are_pruned_and_searched_again(tmp_path: Path):
    kept = routes(tmp_path)
    kept.split(RUN, POWER_ON)
    save_progress(tmp_path / "progress.msgpack", RUN)

    dropped = kept.prune(("Menu", "Play", "Heavy", "Level2"))

    loaded = routes(tmp_path)
    assert dropped == 2
    assert [loaded.segments(part.name) for part in loaded.parts] == [RUN[:1], RUN[1:3], []]
    assert load_progress(tmp_path / "progress.msgpack") == RUN[:3]
    assert loaded.prune(("Menu", "Play", "Heavy", "Level2")) == 0
