from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften.callbacks import save_model_callback as save_model_callback_module
from datenwissenschaften.callbacks.save_model_callback import SaveModelCallback, atomic_save


class _ZipWritingModel:
    def save(self, path: str) -> None:
        Path(path + ".zip").write_bytes(b"model-zip")


class _BareFileWritingModel:
    def save(self, path: str) -> None:
        Path(path).write_bytes(b"model-bare")


class _FailingModel:
    def save(self, path: str) -> None:
        Path(path + ".zip").write_bytes(b"partial-zip")
        Path(path).write_bytes(b"partial-bare")
        raise RuntimeError("save failed")


def test_atomic_save_replaces_target_with_zip_suffixed_save(tmp_path: Path):
    model_path = str(tmp_path / "model")

    atomic_save(_ZipWritingModel(), model_path)

    assert (tmp_path / "model.zip").read_bytes() == b"model-zip"
    assert list(tmp_path.glob("*.tmp*")) == []


def test_atomic_save_replaces_target_when_model_writes_without_zip_suffix(tmp_path: Path):
    model_path = str(tmp_path / "model")

    atomic_save(_BareFileWritingModel(), model_path)

    assert (tmp_path / "model.zip").read_bytes() == b"model-bare"


def test_atomic_save_cleans_up_temp_files_and_reraises_on_failure(tmp_path: Path):
    model_path = str(tmp_path / "model")

    with pytest.raises(RuntimeError, match="save failed"):
        atomic_save(_FailingModel(), model_path)

    assert list(tmp_path.iterdir()) == []


def test_on_step_returns_true():
    assert SaveModelCallback()._on_step() is True


def test_on_rollout_end_saves_checkpoint_and_publishes_metadata(monkeypatch, tmp_path: Path):
    saved = []
    monkeypatch.setattr(save_model_callback_module, "atomic_save", lambda model, path: saved.append((model, path)))
    published = []
    monkeypatch.setattr(
        save_model_callback_module,
        "publish_metadata",
        lambda section, values, *, replace=False: published.append((section, values, replace)),
    )
    model = SimpleNamespace(num_timesteps=42)
    runtime = SimpleNamespace(
        game="Game",
        get_model_path=lambda game: str(tmp_path / f"{game}-model"),
        get_model_metadata=lambda passed_model: {"num_timesteps": passed_model.num_timesteps},
    )
    monkeypatch.setattr(save_model_callback_module, "get_runtime", lambda: runtime)

    callback = SaveModelCallback()
    callback.model = model
    callback.num_timesteps = 42

    result = callback._on_rollout_end()

    assert result is True
    assert saved == [(model, str(tmp_path / "Game-model"))]
    assert published == [("model", {"num_timesteps": 42}, True)]
