from types import SimpleNamespace

from fakes import write_config
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import TrackingSettings
from datenwissenschaften.tracking import run as run_module
from datenwissenschaften.tracking import task_run as task_run_module
from datenwissenschaften.tracking.run import ACTIVE_RUNS, TrackingRun, ending, kill_active_runs, start_tracking
from datenwissenschaften.tracking.task_run import task_run
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.tracking_hook import TrackingHook


class FakeClient:
    def __init__(self, experiments: dict[str, str]) -> None:
        self.experiments = experiments
        self.batches: list[dict] = []
        self.terminated: list[tuple[str, str]] = []
        self.tags: list[tuple[str, str, str]] = []

    def get_experiment_by_name(self, name: str) -> SimpleNamespace | None:
        if name not in self.experiments:
            return None
        return SimpleNamespace(experiment_id=self.experiments[name])

    def create_experiment(self, name: str) -> str:
        self.experiments[name] = "new"
        return "new"

    def create_run(self, experiment_id: str, run_name: str) -> SimpleNamespace:
        return SimpleNamespace(info=SimpleNamespace(run_id=f"{experiment_id}:{run_name}"))

    def log_batch(self, run_id: str, metrics: list = (), params: list = (), synchronous: bool | None = None) -> None:
        self.batches.append({"run": run_id, "metrics": metrics, "params": params, "synchronous": synchronous})

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))


def episode() -> EpisodeRecord:
    return EpisodeRecord(3, "run.bk2", True, "Menu", "Level1", 12.5, 40, 7.0, False, True, False, "Level1")


def test_a_session_is_one_run_in_the_games_experiment_with_its_parameters():
    client = FakeClient({"Game-v0": "7"})

    run = TrackingRun.start(client, "Game-v0", "2026.10.21-1", {"release": "2026.10.21-1"})

    assert run.run_id == "7:2026.10.21-1"
    assert [(param.key, param.value) for param in client.batches[0]["params"]] == [("release", "2026.10.21-1")]


def test_a_new_game_gets_its_own_experiment():
    client = FakeClient({})

    TrackingRun.start(client, "Game-v0", "local", {})

    assert client.experiments == {"Game-v0": "new"}


def test_metrics_are_logged_without_waiting_and_the_run_finishes(monkeypatch):
    flushed = []
    monkeypatch.setattr(run_module, "flush_async_logging", lambda: flushed.append(True))
    client = FakeClient({})
    run = TrackingRun(client, "run")

    run.log({"episode/score": 2.0}, 5)
    run.finish()

    logged = client.batches[0]
    assert [(metric.key, metric.value, metric.step) for metric in logged["metrics"]] == [("episode/score", 2.0, 5)]
    assert logged["synchronous"] is False
    assert flushed == [True]
    assert client.terminated == [("run", "FINISHED")]


def test_without_a_token_nothing_is_tracked():
    assert start_tracking(TrackingSettings("https://training.test", None), "Game-v0", "local", {}) is None


def test_an_unreachable_server_leaves_the_session_untracked(monkeypatch):
    def refuse(cls, client, experiment: str, name: str, parameters: dict[str, str]) -> TrackingRun:
        raise MlflowException("connection refused")

    monkeypatch.setattr(TrackingRun, "start", classmethod(refuse))

    assert start_tracking(TrackingSettings("https://training.test", "token"), "Game-v0", "local", {}) is None


def test_the_hook_logs_every_episode_and_every_update_per_state(monkeypatch):
    logged: list[tuple[dict[str, float], int]] = []
    run = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))
    models = SimpleNamespace(require_active=lambda: "Survive")
    monkeypatch.setattr(
        "datenwissenschaften.training.tracking_hook.learning_metadata",
        lambda models: {"num_timesteps": 512, "entropy_share": 0.5, "explained_variance": 0.25},
    )
    hook = TrackingHook(run, models)

    hook.on_episode_end(episode())
    hook.on_update()

    assert logged[0][1] == 3
    assert logged[0][0]["score/Level1"] == 7.0
    assert logged[0][0]["episode/curriculum_succeeded"] == 1.0
    assert logged[1] == ({"Survive/entropy_share": 0.5, "Survive/explained_variance": 0.25}, 512)


def test_a_failed_run_keeps_its_error_and_ends_failed(monkeypatch):
    monkeypatch.setattr(run_module, "flush_async_logging", lambda: None)
    client = FakeClient({})

    TrackingRun(client, "run").fail(ValueError("no ROM"))

    assert client.tags == [("run", "error", "ValueError: no ROM")]
    assert client.terminated == [("run", "FAILED")]


def test_every_task_is_a_run_in_the_games_task_experiment_that_ends_with_the_task(tmp_path, monkeypatch):
    started, ended = [], []
    run = SimpleNamespace(end=lambda error: ended.append("FINISHED" if error is None else str(error)))

    def start(settings, experiment: str, name: str, parameters: dict[str, str]) -> SimpleNamespace:
        started.append((experiment, name, parameters))
        return run

    monkeypatch.setattr(task_run_module, "start_tracking", start)
    config = write_config(tmp_path)

    with task_run(config, "train_session"):
        pass
    try:
        with task_run(config, "ab_test"):
            raise RuntimeError("emulator crashed")
    except RuntimeError:
        pass

    assert started == [
        ("FakeGame-v0 tasks", "train_session", {"release": "local"}),
        ("FakeGame-v0 tasks", "ab_test", {"release": "local"}),
    ]
    assert ended == ["FINISHED", "emulator crashed"]


def test_an_untracked_task_still_runs(tmp_path):
    ran = []

    with task_run(write_config(tmp_path), "lab_run"):
        ran.append(True)

    assert ran == [True]


def test_a_run_ends_finished_killed_or_failed_by_how_its_work_ended(monkeypatch):
    monkeypatch.setattr(run_module, "flush_async_logging", lambda: None)
    client = FakeClient({})

    TrackingRun(client, "done").end(None)
    TrackingRun(client, "stopped").end(KeyboardInterrupt())
    TrackingRun(client, "exited").end(SystemExit(143))
    TrackingRun(client, "broken").end(ValueError("no ROM"))

    assert client.terminated == [
        ("done", "FINISHED"),
        ("stopped", "KILLED"),
        ("exited", "KILLED"),
        ("broken", "FAILED"),
    ]
    assert client.tags == [("broken", "error", "ValueError: no ROM")]


def test_open_runs_are_killed_when_the_container_stops(monkeypatch):
    monkeypatch.setattr(run_module, "flush_async_logging", lambda: None)
    client = FakeClient({})
    finished = TrackingRun.start(client, "Game-v0", "earlier", {})
    finished.finish()
    open_run = TrackingRun.start(client, "Game-v0", "session", {})

    kill_active_runs()

    assert client.terminated == [(finished.run_id, "FINISHED"), (open_run.run_id, "KILLED")]
    assert not ACTIVE_RUNS


def test_ending_ends_the_run_by_how_the_block_exits_and_passes_without_one(monkeypatch):
    ended: list[BaseException | None] = []
    run = SimpleNamespace(end=ended.append)

    with ending(run):
        pass
    try:
        with ending(run):
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    with ending(None):
        pass

    assert ended[0] is None
    assert isinstance(ended[1], KeyboardInterrupt)
