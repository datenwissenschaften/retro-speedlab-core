from pathlib import Path

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.training import rotation as rotation_module
from datenwissenschaften.training.rotation import Rotation

LEVELS = ("Level1", "Level2", "Level3")


def _rotation(tmp_path: Path) -> Rotation:
    return Rotation(JsonDatabase(tmp_path / "database.json"), "Game", LEVELS, 60)


def test_levels_take_turns_and_wrap_around(tmp_path: Path, monkeypatch):
    clock = iter([0, 3600, 7200, 10800])
    monkeypatch.setattr(rotation_module.time, "time", lambda: next(clock))
    rotation = _rotation(tmp_path)

    assert [rotation.next() for _ in range(4)] == [(level, 3600) for level in (*LEVELS, "Level1")]


def test_a_restart_continues_the_running_level_with_the_time_left(tmp_path: Path, monkeypatch):
    clock = iter([0, 1000])
    monkeypatch.setattr(rotation_module.time, "time", lambda: next(clock))
    _rotation(tmp_path).next()

    assert _rotation(tmp_path).next() == ("Level1", 2600)


def test_a_removed_level_starts_the_rotation_over(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rotation_module.time, "time", lambda: 0)
    database = JsonDatabase(tmp_path / "database.json")
    database.set("rotation:Game", {"savestate": "Level9", "ends_at": 100})

    assert Rotation(database, "Game", LEVELS, 60).next() == ("Level1", 3600)
