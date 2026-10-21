import signal
from pathlib import Path

import pytest

from datenwissenschaften.workflow import launch


class Exited(Exception):
    pass


def test_a_stopped_container_kills_the_open_runs_and_exits_at_once(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(launch, "kill_active_runs", lambda: calls.append("killed"))

    def exit_now(status: int) -> None:
        calls.append(f"exit {status}")
        raise Exited

    monkeypatch.setattr(launch.os, "_exit", exit_now)

    with pytest.raises(Exited):
        launch.terminate(signal.SIGTERM, None)

    assert calls == ["killed", "exit 143"]


def test_training_starts_after_the_stop_handler_is_installed(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(launch.signal, "signal", lambda number, handler: calls.append(("signal", number, handler)))
    monkeypatch.setattr(launch.flyte, "init", lambda: calls.append(("init",)))
    monkeypatch.setattr(launch.flyte, "run", lambda task, **inputs: calls.append(("run", task, inputs)))

    launch.run_training("src.game.wrapper:GameWrapper", Path("config.yaml"))

    assert calls == [
        ("signal", signal.SIGTERM, launch.terminate),
        ("init",),
        ("run", launch.training, {"wrapper": "src.game.wrapper:GameWrapper", "config": "config.yaml"}),
    ]
