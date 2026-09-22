from pathlib import Path

import pytest

from datenwissenschaften.ui import control as control_module
from datenwissenschaften.ui.control import (
    TrainingControl,
    configure_training_control,
    consume_model_reset,
    control_metadata,
    model_reset_requested,
    request_model_reset,
)


def test_configure_sets_defaults_and_resets_any_pending_request(tmp_path: Path):
    control = TrainingControl()
    control.configure(game="Game", model_dir=tmp_path, restart_supported=True)

    assert control.metadata() == {"game": "Game", "restart_supported": True, "reset_pending": False}


def test_request_reset_requires_restart_support(tmp_path: Path):
    control = TrainingControl()
    control.configure(game="Game", model_dir=tmp_path, restart_supported=False)

    with pytest.raises(RuntimeError, match="does not support an in-process restart"):
        control.request_reset("Game")


def test_request_reset_requires_a_configured_game(tmp_path: Path):
    control = TrainingControl()

    with pytest.raises(RuntimeError, match="does not support an in-process restart"):
        control.request_reset("Game")


def test_request_reset_rejects_a_mismatched_game(tmp_path: Path):
    control = TrainingControl()
    control.configure(game="Game", model_dir=tmp_path, restart_supported=True)

    with pytest.raises(ValueError, match="does not match the active training run"):
        control.request_reset("OtherGame")


def test_request_reset_returns_the_same_pending_request_until_consumed(tmp_path: Path):
    control = TrainingControl()
    control.configure(game="Game", model_dir=tmp_path, restart_supported=True)

    first = control.request_reset("Game")
    second = control.request_reset("Game")

    assert first is second
    assert control.reset_requested() is True

    consumed = control.consume_reset()

    assert consumed is first
    assert control.reset_requested() is False
    assert control.consume_reset() is None


def test_module_level_helpers_delegate_to_the_shared_control(monkeypatch, tmp_path: Path):
    control = TrainingControl()
    monkeypatch.setattr(control_module, "_control", control)

    cleanup = []
    configure_training_control(
        game="Game",
        model_dir=tmp_path,
        restart_supported=True,
        artifact_dirs=(tmp_path,),
        on_reset=lambda: cleanup.append(True),
    )

    assert control_metadata() == {"game": "Game", "restart_supported": True, "reset_pending": False}
    assert model_reset_requested() is False

    request = request_model_reset("Game")

    assert model_reset_requested() is True
    assert request.artifact_dirs == (tmp_path.resolve(),)

    consumed = consume_model_reset()

    assert consumed is request
    assert model_reset_requested() is False
