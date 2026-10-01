from pathlib import Path

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.training import rotation as rotation_module
from datenwissenschaften.training.rotation import BEATEN_FULL_RUN_WINS, Rotation

LEVELS = ("Level1", "Level2", "Level3")
ALL_BEATEN = {level: BEATEN_FULL_RUN_WINS for level in LEVELS}


def _rotation(tmp_path: Path, wins: dict[str, int]) -> Rotation:
    return Rotation(JsonDatabase(tmp_path / "database.json"), "Game", LEVELS, 120, wins.__getitem__)


def test_each_level_is_trained_until_beaten_before_the_next(tmp_path: Path):
    wins = {"Level1": BEATEN_FULL_RUN_WINS, "Level2": BEATEN_FULL_RUN_WINS - 1, "Level3": 0}

    assert _rotation(tmp_path, wins).next() == ("Level2", 7200, False)


def test_beaten_levels_take_turns_as_speedruns_and_wrap_around(tmp_path: Path, monkeypatch):
    clock = iter([0, 7200, 14400, 21600])
    monkeypatch.setattr(rotation_module.time, "time", lambda: next(clock))
    rotation = _rotation(tmp_path, ALL_BEATEN)

    assert [rotation.next() for _ in range(4)] == [(level, 7200, True) for level in (*LEVELS, "Level1")]


def test_a_restart_continues_the_running_speedrun_level_with_the_time_left(tmp_path: Path, monkeypatch):
    clock = iter([0, 1000])
    monkeypatch.setattr(rotation_module.time, "time", lambda: next(clock))
    _rotation(tmp_path, ALL_BEATEN).next()

    assert _rotation(tmp_path, ALL_BEATEN).next() == ("Level1", 6200, True)


def test_a_removed_level_starts_the_speedrun_rotation_over(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rotation_module.time, "time", lambda: 0)
    database = JsonDatabase(tmp_path / "database.json")
    database.set("rotation:Game", {"savestate": "Level9", "ends_at": 100})

    assert Rotation(database, "Game", LEVELS, 120, ALL_BEATEN.__getitem__).next() == ("Level1", 7200, True)
