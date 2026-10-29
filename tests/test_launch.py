from pathlib import Path

from datenwissenschaften.workflow import launch
from datenwissenschaften.workflow.speedrun import speedrun


def test_a_task_runs_after_flyte_starts(monkeypatch):
    calls: list[tuple] = []
    monkeypatch.setattr(launch.flyte, "init", lambda: calls.append(("init",)))
    monkeypatch.setattr(launch.flyte, "run", lambda task, **inputs: calls.append(("run", task, inputs)))

    launch.launch(speedrun, "src.game.wrapper:GameWrapper", Path("config.yaml"))

    assert calls == [
        ("init",),
        ("run", speedrun, {"wrapper": "src.game.wrapper:GameWrapper", "config": "config.yaml"}),
    ]
