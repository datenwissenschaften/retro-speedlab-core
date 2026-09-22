from types import SimpleNamespace

from datenwissenschaften.callbacks import episode_telemetry_callback as episode_telemetry_callback_module
from datenwissenschaften.callbacks.episode_telemetry_callback import EpisodeTelemetryCallback


class _FakeVecEnv:
    def __init__(self, num_envs: int, method_results: dict):
        self.num_envs = num_envs
        self._method_results = method_results

    def env_method(self, name: str):
        result = self._method_results.get(name)
        if isinstance(result, Exception):
            raise result
        return result


def _with_env(callback: EpisodeTelemetryCallback, env: _FakeVecEnv, *, has_rollout_buffer: bool = False) -> None:
    model = SimpleNamespace(get_env=lambda: env)
    if has_rollout_buffer:
        model.rollout_buffer = object()
    callback.model = model


def test_on_training_start_initializes_per_env_state_when_rollout_buffer_present():
    callback = EpisodeTelemetryCallback()
    _with_env(
        callback,
        _FakeVecEnv(2, {"episode_start_state": ["Find", "Eat"]}),
        has_rollout_buffer=True,
    )

    callback._on_training_start()

    assert callback.enabled is True
    assert callback.started_at != []
    assert callback.fitness == [0.0, 0.0]
    assert callback.steps == [0, 0]
    assert callback.training_states == ["Find", "Eat"]


def test_on_training_start_disabled_when_model_has_no_rollout_buffer():
    callback = EpisodeTelemetryCallback()
    _with_env(callback, _FakeVecEnv(2, {}))

    callback._on_training_start()

    assert callback.enabled is False
    assert callback.started_at == []
    assert callback.fitness == []
    assert callback.steps == []
    assert callback.training_states == []


def test_on_step_returns_true_immediately_when_disabled():
    callback = EpisodeTelemetryCallback()
    callback.enabled = False

    assert callback._on_step() is True


def test_on_step_returns_true_when_locals_are_incomplete():
    callback = EpisodeTelemetryCallback()
    callback.enabled = True
    callback.locals = {"rewards": None, "dones": None, "infos": None}

    assert callback._on_step() is True


def test_on_step_publishes_episode_and_resets_tracking_on_done(monkeypatch):
    published = []
    monkeypatch.setattr(
        episode_telemetry_callback_module,
        "publish_episode",
        lambda **kwargs: published.append(kwargs),
    )
    monkeypatch.setattr(
        episode_telemetry_callback_module,
        "get_runtime",
        lambda: SimpleNamespace(savestate="Level1"),
    )

    callback = EpisodeTelemetryCallback()
    callback.enabled = True
    _with_env(callback, _FakeVecEnv(2, {"episode_start_state": ["A", "B"]}))
    callback.started_at = [0.0, 0.0]
    callback.fitness = [0.0, 0.0]
    callback.steps = [0, 0]
    callback.training_states = [None, None]
    callback.locals = {
        "rewards": [1.0, 2.0],
        "dones": [True, False],
        "infos": [
            {
                "extrinsic_reward": 1.0,
                "won": True,
                "episode": {"r": 5.0, "l": 10},
                "started_from_initial_savestate": True,
                "state": "Find",
            },
            {"extrinsic_reward": 2.0},
        ],
    }

    result = callback._on_step()

    assert result is True
    assert len(published) == 1
    assert published[0]["env"] == 0
    assert published[0]["savestate"] == "Level1"
    assert published[0]["fitness"] == 5.0
    assert published[0]["training_steps"] == 10
    assert published[0]["won"] is True
    assert published[0]["started_from_initial_savestate"] is True
    assert callback.fitness[0] == 0.0
    assert callback.steps[0] == 0
    assert callback.training_states[0] == "A"
    assert callback.training_states[1] is None
    assert callback.fitness[1] == 2.0


def test_on_step_does_not_refresh_training_states_when_nothing_is_done():
    callback = EpisodeTelemetryCallback()
    callback.enabled = True
    _with_env(callback, _FakeVecEnv(1, {"episode_start_state": AssertionError()}))
    callback.started_at = [0.0]
    callback.fitness = [0.0]
    callback.steps = [0]
    callback.training_states = [None]
    callback.locals = {
        "rewards": [1.0],
        "dones": [False],
        "infos": [{"extrinsic_reward": 1.0}],
    }

    result = callback._on_step()

    assert result is True
    assert callback.fitness[0] == 1.0
    assert callback.training_states == [None]


def test_state_names_falls_back_to_state_name_when_episode_start_state_unavailable():
    callback = EpisodeTelemetryCallback()
    _with_env(callback, _FakeVecEnv(2, {"episode_start_state": AttributeError(), "state_name": ["Boss"]}))

    assert callback._state_names(2) == ["Boss", None]


def test_state_names_returns_none_list_when_both_probes_fail():
    callback = EpisodeTelemetryCallback()
    _with_env(callback, _FakeVecEnv(1, {"episode_start_state": TypeError(), "state_name": ValueError()}))

    assert callback._state_names(1) == [None]


def test_state_names_returns_none_list_when_probe_returns_nothing():
    callback = EpisodeTelemetryCallback()
    _with_env(callback, _FakeVecEnv(0, {"episode_start_state": None}))

    assert callback._state_names(0) == []
