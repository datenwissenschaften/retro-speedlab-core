import sys
from pathlib import Path

from datenwissenschaften import roms


def test_import_roms_runs_the_stable_retro_importer(monkeypatch, tmp_path: Path):
    calls = []
    monkeypatch.setattr(roms.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))

    roms.import_roms(tmp_path)

    command, kwargs = calls[0]
    assert command == [sys.executable, "-m", "stable_retro.import", str(tmp_path)]
    assert kwargs["check"] is True


def test_extra_savestates_are_copied_next_to_the_game_integration(monkeypatch, tmp_path: Path):
    integration, extra = tmp_path / "FakeGame-v0", tmp_path / "savestates"
    integration.mkdir()
    extra.mkdir()
    (extra / "Level2.state").write_bytes(b"level two")
    (extra / "notes.txt").write_text("ignored")
    monkeypatch.setattr(roms.stable_retro.data, "get_file_path", lambda game, name: str(integration / name))

    roms.import_savestates("FakeGame-v0", extra)

    assert (integration / "Level2.state").read_bytes() == b"level two"
    assert not (integration / "notes.txt").exists()
