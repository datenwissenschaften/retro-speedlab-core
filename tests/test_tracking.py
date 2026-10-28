from types import SimpleNamespace

from fakes import write_config
from mlflow.entities import RunStatus
from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import TrackingSettings
from datenwissenschaften.tracking import task_run as task_run_module
from datenwissenschaften.tracking.run import ACTIVE_RUNS, TrackingRun, ending, kill_active_runs, start_tracking
from datenwissenschaften.tracking.task_run import task_run
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.tracking_hook import TrackingHook, chart_group


class FakeClient:
    def __init__(self, experiments: dict[str, str]) -> None:
        self.experiments = experiments
        self.batches: list[dict] = []
        self.terminated: list[tuple[str, str]] = []
        self.tags: list[tuple[str, str, str]] = []
        self.flushes = 0
        store = SimpleNamespace(update_run_info=self.update_run_info, flush_async_logging=self.flush_async_logging)
        self._tracking_client = SimpleNamespace(store=store)

    def flush_async_logging(self) -> None:
        self.flushes += 1

    def update_run_info(self, run_id: str, status: int, end_time: int, run_name: None) -> None:
        self.terminated.append((run_id, RunStatus.to_string(status)))

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


def test_metrics_are_logged_without_waiting_and_the_run_finishes_after_flushing_its_own_server():
    client = FakeClient({})
    run = TrackingRun(client, "run")

    run.log({"episode/score": 2.0}, 5)
    run.finish()

    logged = client.batches[0]
    assert [(metric.key, metric.value, metric.step) for metric in logged["metrics"]] == [("episode/score", 2.0, 5)]
    assert logged["synchronous"] is False
    assert client.flushes == 1
    assert client.terminated == [("run", "FINISHED")]


def test_without_a_token_nothing_is_tracked():
    assert start_tracking(TrackingSettings("https://training.test", None), "Game-v0", "local", {}) is None


def test_an_unreachable_server_leaves_the_session_untracked(monkeypatch):
    def refuse(cls, client, experiment: str, name: str, parameters: dict[str, str]) -> TrackingRun:
        raise MlflowException("connection refused")

    monkeypatch.setattr(TrackingRun, "start", classmethod(refuse))

    assert start_tracking(TrackingSettings("https://training.test", "token"), "Game-v0", "local", {}) is None


def test_the_hook_logs_each_levels_attempts_and_the_overall_speed_and_confidence(monkeypatch):
    logged: list[tuple[dict[str, float], int]] = []
    run = SimpleNamespace(log=lambda metrics, step: logged.append((metrics, step)))
    monkeypatch.setattr(
        "datenwissenschaften.training.tracking_hook.learning_metadata",
        lambda models: {"num_timesteps": 512, "entropy_share": 0.25, "explained_variance": 0.5},
    )
    monkeypatch.setattr("datenwissenschaften.training.tracking_hook.resident_megabytes", lambda: 900.0)
    hook = TrackingHook(run, SimpleNamespace(), ("Menu", "Level1", "Boss"), {"Level 1": ("Level1",)})

    hook.on_episode_end(episode())
    hook.on_episode_end(episode())
    hook.on_update()

    assert logged == [
        ({"Level 1/Level1/progress": 1.0, "Level 1/Level1/success": 1.0, "Level 1/Level1/score": 7.0}, 1),
        ({"resident_memory_mb": 900.0, "decisions_per_second": 3.2}, 3),
        ({"Level 1/Level1/progress": 1.0, "Level 1/Level1/success": 1.0, "Level 1/Level1/score": 7.0}, 2),
        ({"resident_memory_mb": 900.0, "decisions_per_second": 3.2}, 3),
        ({"confidence": 0.75}, 1),
    ]


def test_an_attempt_without_duration_has_no_speed(monkeypatch):
    monkeypatch.setattr("datenwissenschaften.training.tracking_hook.resident_megabytes", lambda: 900.0)
    logged: list[dict[str, float]] = []
    run = SimpleNamespace(log=lambda metrics, step: logged.append(metrics))
    record = episode()
    record.duration_seconds = 0.0

    TrackingHook(run, SimpleNamespace(), ("Menu", "Level1"), {"Level 1": ("Level1",)}).on_episode_end(record)

    assert logged == [
        {"Level 1/Level1/progress": 1.0, "Level 1/Level1/success": 1.0, "Level 1/Level1/score": 7.0},
        {"resident_memory_mb": 900.0},
    ]


def test_a_failed_run_keeps_its_error_and_ends_failed():
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


def test_a_run_ends_finished_killed_or_failed_by_how_its_work_ended():
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


def test_open_runs_are_killed_when_the_container_stops():
    client = FakeClient({})
    finished = TrackingRun.start(client, "Game-v0", "earlier", {})
    finished.finish()
    open_run = TrackingRun.start(client, "Game-v0", "session", {})

    kill_active_runs()

    assert client.terminated == [(finished.run_id, "FINISHED"), (open_run.run_id, "KILLED")]
    assert not ACTIVE_RUNS


def test_ending_ends_the_run_by_how_the_block_exits_and_passes_without_one():
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


def test_charts_group_every_part_under_its_level_and_full_runs_on_their_own():
    levels = {"Level 1": ("Grow", "Scale"), "Level 3": ("Jump3",)}

    assert chart_group("Jump3", levels) == "Level 3/Jump3"
    assert chart_group("Level 1", levels) == "Level 1/whole level"
    assert chart_group("Full run", levels) == "Full run"
