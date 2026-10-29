from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften.tas.levels import LevelTrainer
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.routes import FULL_RUN, ROUTES_FILE, Leg, Part, Routes, load_progress, parts_of
from datenwissenschaften.tas.runner import Start

COURSE = SimpleNamespace(first="Menu", levels={"Level 1": ("Play", "Heavy"), "Level 2": ("Level2", "Boss2")})
POWER_ON = Start("Menu", b"on")
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


def test_the_full_run_is_its_own_movie(tmp_path: Path):
    kept = routes(tmp_path)
    kept.save(Leg(FULL_RUN, POWER_ON, None), RUN)

    assert load_progress(tmp_path / "progress.msgpack") == RUN
    assert not (tmp_path / ROUTES_FILE).exists()


def test_the_trainer_beats_every_part_then_retrains_each_and_only_then_builds_the_full_run(tmp_path: Path):
    kept = routes(tmp_path)
    kept.save(Leg(FULL_RUN, POWER_ON, None), [*RUN[:3], Segment("Boss2", [[1]], b"won", True)])
    calls: list[tuple[str, str, int]] = []

    def extend(leg: Leg, segments: list[Segment]) -> list[Segment]:
        calls.append(("extend", leg.name, len(segments)))
        return segments

    def improve(leg: Leg, segments: list[Segment]) -> list[Segment]:
        calls.append(("improve", leg.name, len(segments)))
        if leg.name == FULL_RUN:
            raise OneRound
        return segments

    trainer = LevelTrainer(SimpleNamespace(extend=extend, improve=improve), kept, POWER_ON)
    with pytest.raises(OneRound):
        trainer.forever()

    assert calls == [
        ("extend", "Menu", 1),
        ("extend", "Level 1", 2),
        ("extend", "Level 2", 1),
        ("improve", "Menu", 1),
        ("improve", "Level 1", 2),
        ("improve", "Level 2", 1),
        ("extend", FULL_RUN, 4),
        ("improve", FULL_RUN, 4),
    ]
    assert load_progress(tmp_path / "progress.msgpack") == []
