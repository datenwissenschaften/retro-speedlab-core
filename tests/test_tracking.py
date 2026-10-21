from types import SimpleNamespace

from mlflow.exceptions import MlflowException

from datenwissenschaften.settings import TrackingSettings
from datenwissenschaften.tracking import run as run_module
from datenwissenschaften.tracking.run import TrackingRun, start_tracking
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.tracking_hook import TrackingHook


class FakeClient:
    def __init__(self, experiments: dict[str, str]) -> None:
        self.experiments = experiments
        self.batches: list[dict] = []
        self.terminated: list[tuple[str, str]] = []

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
