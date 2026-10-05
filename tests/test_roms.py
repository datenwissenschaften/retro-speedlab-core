from pathlib import Path

from datenwissenschaften import roms


def test_roms_are_imported_into_the_stable_and_the_projects_own_integrations(monkeypatch, tmp_path: Path):
    calls = []
    (tmp_path / "roms").mkdir()
    (tmp_path / "roms" / "game.nes").write_bytes(b"rom")
    monkeypatch.setattr(roms.retro.data, "add_custom_integration", lambda path: calls.append(("custom", path)))
    monkeypatch.setattr(roms.retro.data, "merge", lambda *paths, quiet: calls.append(("merge", paths)))

    roms.import_roms(tmp_path / "roms", tmp_path / "integrations")

    assert calls == [("custom", str(tmp_path / "integrations")), ("merge", (str(tmp_path / "roms" / "game.nes"),))]
