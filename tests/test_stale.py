from pathlib import Path
from types import SimpleNamespace

from fakes import write_config
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import load_config
from datenwissenschaften.tracking import stale


class FakeClient:
    def __init__(self, runs: dict[str, list[str]]) -> None:
        self.runs = runs
        self.terminated: list[tuple[str, str]] = []

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        return SimpleNamespace(experiment_id=name) if name in self.runs else None

    def search_runs(self, experiment_ids: list[str], filter_string: str) -> list[SimpleNamespace]:
        assert filter_string == stale.RUNNING
        names = self.runs[experiment_ids[0]]
        return [SimpleNamespace(info=SimpleNamespace(run_id=f"id-{name}", run_name=name)) for name in names]

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))


def tracked_config(tmp_path: Path):
    path = write_config(tmp_path)
    path.write_text(path.read_text().replace("token: null", "token: secret"), encoding="utf-8")
    return load_config(path)


def test_a_new_session_closes_every_open_session_and_training_task(tmp_path: Path, monkeypatch):
    client = FakeClient(
        {
            "FakeGame-v0": ["2026.10.21-8", "2026.10.21-9"],
            "FakeGame-v0 tasks": ["train_session", "lab_run"],
        }
    )
    monkeypatch.setattr(stale, "tracking_client", lambda settings: client)

    stale.close_stale_training_runs(tracked_config(tmp_path))

    assert client.terminated == [
        ("id-2026.10.21-8", "KILLED"),
        ("id-2026.10.21-9", "KILLED"),
        ("id-train_session", "KILLED"),
    ]


def test_without_a_running_lab_only_open_lab_runs_are_closed(tmp_path: Path, monkeypatch):
    client = FakeClient({"FakeGame-v0 tasks": ["train_session", "lab_run"]})
    monkeypatch.setattr(stale, "tracking_client", lambda settings: client)

    stale.close_stale_lab_runs(tracked_config(tmp_path))

    assert client.terminated == [("id-lab_run", "KILLED")]


def test_a_game_without_experiments_or_tracking_closes_nothing(tmp_path: Path, monkeypatch):
    client = FakeClient({})
    monkeypatch.setattr(stale, "tracking_client", lambda settings: client)

    stale.close_stale_training_runs(tracked_config(tmp_path))
    stale.close_stale_lab_runs(load_config(write_config(tmp_path)))

    assert client.terminated == []


def test_an_unreachable_server_leaves_the_runs_open_without_stopping_training(tmp_path: Path, monkeypatch):
    class Unreachable(FakeClient):
        def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
            raise MlflowException("connection refused")

    monkeypatch.setattr(stale, "tracking_client", lambda settings: Unreachable({}))

    stale.close_stale_training_runs(tracked_config(tmp_path))
