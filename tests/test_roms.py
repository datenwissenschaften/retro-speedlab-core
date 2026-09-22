from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften import roms as roms_module


def test_import_roms_resolves_the_roms_path_from_config_when_not_given(monkeypatch):
    calls = []
    monkeypatch.setattr(
        roms_module,
        "load_config",
        lambda config_path: SimpleNamespace(paths=SimpleNamespace(roms_path=Path("/configured/roms"))),
    )
    monkeypatch.setattr(roms_module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    roms_module.import_roms(None, config_path="config.yaml")

    (call_args, call_kwargs) = calls[0]
    assert str(Path("/configured/roms")) in call_args[0]
    assert call_kwargs["check"] is True


def test_import_roms_uses_an_explicit_roms_directory(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(roms_module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    roms_module.import_roms(tmp_path, config_path="config.yaml")

    (call_args, _call_kwargs) = calls[0]
    assert str(tmp_path) in call_args[0]
