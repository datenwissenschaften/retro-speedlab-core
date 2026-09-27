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
