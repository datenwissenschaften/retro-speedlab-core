import json
import time

import pytest
from redis.exceptions import RedisError

from datenwissenschaften.ui import telemetry as telemetry_module
from datenwissenschaften.ui.telemetry import TelemetryStore


class FakeRedisClient:
    instances: list["FakeRedisClient"] = []

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ping_error: Exception | None = None
        self.set_error: Exception | None = None
        self.delete_error: Exception | None = None
        FakeRedisClient.instances.append(self)

    @classmethod
    def from_url(cls, redis_url, **kwargs):
        client = cls()
        client.redis_url = redis_url
        return client

    def ping(self):
        if self.ping_error is not None:
            raise self.ping_error

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        if self.set_error is not None:
            raise self.set_error
        self.store[key] = value

    def delete(self, key):
        if self.delete_error is not None:
            raise self.delete_error
        self.store.pop(key, None)


def test_publish_episode_and_metadata_update_the_snapshot():
    store = TelemetryStore()

    store.publish_episode({"won": True, "training_state": "Explore", "fitness": 4.0})
    store.publish_metadata("run", {"game": "Game"})

    snapshot = store.snapshot()

    assert snapshot["summary"]["episodes"] == 1
    assert snapshot["summary"]["wins"] == 1
    assert snapshot["summary"]["by_state"]["Explore"]["episodes"] == 1
    assert snapshot["metadata"]["run"] == {"game": "Game"}


def test_publish_episode_tracks_full_run_savestate_and_final_state_details():
    store = TelemetryStore()

    store.publish_episode(
        {
            "won": True,
            "training_state": "Explore",
            "savestate": "Level1",
            "fitness": 4.0,
            "duration_seconds": 2.5,
            "started_from_initial_savestate": True,
            "final_state": "Explore",
        }
    )
    store.publish_episode(
        {
            "won": True,
            "training_state": "Explore",
            "savestate": "Level1",
            "fitness": 9.0,
            "duration_seconds": 1.0,
            "started_from_initial_savestate": True,
            "final_state": "Explore",
        }
    )

    summary = store.snapshot()["summary"]
    assert summary["full_run_episodes"] == 2
    assert summary["full_run_wins"] == 2
    assert summary["full_run_timed_episodes"] == 2
    assert summary["full_run_best_fitness"] == 9.0
    assert summary["latest_duration_seconds"] == 1.0
    assert summary["latest_final_state"] == "Explore"
    assert summary["by_savestate"]["Level1"]["episodes"] == 2
    assert summary["by_state"]["Explore"]["full_run_episodes"] == 2


def test_publish_metadata_merges_by_default_and_replaces_when_asked():
    store = TelemetryStore()

    store.publish_metadata("run", {"a": 1})
    store.publish_metadata("run", {"b": 2})
    assert store.snapshot()["metadata"]["run"] == {"a": 1, "b": 2}

    store.publish_metadata("run", {"c": 3}, replace=True)
    assert store.snapshot()["metadata"]["run"] == {"c": 3}


def test_clear_metadata_optionally_resets_history():
    store = TelemetryStore()
    store.publish_episode({"won": True})
    store.publish_metadata("run", {"a": 1})

    store.clear_metadata("run", clear_history=True)

    snapshot = store.snapshot()
    assert "run" not in snapshot["metadata"]
    assert snapshot["summary"]["episodes"] == 0


def test_clear_metadata_is_a_noop_for_an_unknown_section():
    store = TelemetryStore()

    store.clear_metadata("missing")

    assert store.snapshot()["metadata"] == {}


def test_configure_history_requires_the_redis_package(monkeypatch):
    monkeypatch.setattr(telemetry_module, "Redis", None)
    store = TelemetryStore()

    with pytest.raises(RuntimeError, match="requires the 'redis' package"):
        store.configure_history("Game")


def test_configure_history_raises_when_redis_is_unreachable(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)

    def failing_from_url(redis_url, **kwargs):
        client = FakeRedisClient()
        client.ping_error = RedisError("boom")
        return client

    monkeypatch.setattr(FakeRedisClient, "from_url", staticmethod(failing_from_url))
    store = TelemetryStore()

    with pytest.raises(RuntimeError, match="Could not connect to Redis"):
        store.configure_history("Game", redis_url="redis://example")


def test_configure_history_is_idempotent_for_the_same_scope(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    first_client = FakeRedisClient.instances[-1]
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert len(FakeRedisClient.instances) == 1
    assert store._redis is first_client


def test_configure_history_loads_a_valid_persisted_summary(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    payload = {
        "started_at": "2024-01-01T00:00:00+00:00",
        "metadata": {"run": {"game": "Game"}},
        "summary": {
            "episodes": 5,
            "wins": 2,
            "full_run_episodes": 3,
            "full_run_wins": 1,
            "timed_episodes": 4,
            "full_run_timed_episodes": 2,
            "duration_seconds_total": 12.5,
            "full_run_duration_seconds_total": 6.0,
            "best_fitness": 9.5,
            "full_run_best_fitness": 8.0,
            "latest_index": 5,
            "latest_timestamp": "2024-01-01T00:01:00+00:00",
            "latest_training_state": "Explore",
            "latest_duration_seconds": 1.5,
            "latest_full_run_duration_seconds": 1.0,
            "latest_final_state": "Explore",
            "by_state": {
                "Explore": {"episodes": 5, "wins": 2},
                "": {"episodes": 1},
                "bad": "not-a-dict",
            },
            "by_savestate": {"Level1": {"episodes": 5, "wins": 2}},
        },
    }

    def from_url(redis_url, **kwargs):
        client = FakeRedisClient()
        client.store["prefix:Game"] = json.dumps(payload)
        return client

    monkeypatch.setattr(FakeRedisClient, "from_url", staticmethod(from_url))
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    snapshot = store.snapshot()
    assert snapshot["metadata"]["run"] == {"game": "Game"}
    assert snapshot["summary"]["episodes"] == 5
    assert snapshot["summary"]["by_state"]["Explore"]["wins"] == 2
    assert "" not in snapshot["summary"]["by_state"]
    assert "bad" not in snapshot["summary"]["by_state"]
    assert snapshot["summary"]["by_savestate"]["Level1"]["episodes"] == 5


def test_configure_history_ignores_malformed_persisted_history(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)

    def from_url(redis_url, **kwargs):
        client = FakeRedisClient()
        client.store["prefix:Game"] = "not-json"
        return client

    monkeypatch.setattr(FakeRedisClient, "from_url", staticmethod(from_url))
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert store.snapshot()["summary"]["episodes"] == 0


def test_configure_history_ignores_history_with_wrong_metadata_type(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)

    def from_url(redis_url, **kwargs):
        client = FakeRedisClient()
        client.store["prefix:Game"] = json.dumps({"metadata": "not-a-dict", "summary": {}})
        return client

    monkeypatch.setattr(FakeRedisClient, "from_url", staticmethod(from_url))
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert store.snapshot()["metadata"] == {}


def test_configure_history_falls_back_to_an_empty_summary_when_summary_is_not_a_dict(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)

    def from_url(redis_url, **kwargs):
        client = FakeRedisClient()
        client.store["prefix:Game"] = json.dumps({"metadata": {}, "summary": "not-a-dict"})
        return client

    monkeypatch.setattr(FakeRedisClient, "from_url", staticmethod(from_url))
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert store.snapshot()["summary"]["episodes"] == 0


def test_configure_history_handles_no_persisted_history(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()

    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert store.snapshot()["summary"]["episodes"] == 0


def test_flush_persists_the_current_snapshot_to_redis(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    store.publish_episode({"won": True})

    store.flush()

    client = FakeRedisClient.instances[-1]
    persisted = json.loads(client.store["prefix:Game"])
    assert persisted["summary"]["episodes"] == 1


def test_flush_is_a_noop_when_history_is_not_configured():
    store = TelemetryStore()

    store.flush()


def test_flush_skips_when_expected_version_is_stale(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    store.flush(expected_version=store._history_version - 1)

    client = FakeRedisClient.instances[-1]
    assert "prefix:Game" not in client.store


def test_flush_logs_and_swallows_redis_errors(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    client = FakeRedisClient.instances[-1]
    client.set_error = RedisError("boom")

    store.flush()


def test_reset_for_restart_clears_local_and_redis_state(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    store.publish_episode({"won": True})
    store.flush()
    client = FakeRedisClient.instances[-1]
    assert "prefix:Game" in client.store

    cleanup_calls = []
    store.reset_for_restart(lambda: cleanup_calls.append(True))

    assert cleanup_calls == [True]
    assert "prefix:Game" not in client.store
    assert store.snapshot()["summary"]["episodes"] == 0


def test_reset_for_restart_without_redis_configured_still_runs_cleanup():
    store = TelemetryStore()

    cleanup_calls = []
    store.reset_for_restart(lambda: cleanup_calls.append(True))

    assert cleanup_calls == [True]


def test_reset_for_restart_logs_and_swallows_redis_delete_errors(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    client = FakeRedisClient.instances[-1]
    client.delete_error = RedisError("boom")

    store.reset_for_restart(lambda: None)


def test_resize_is_a_noop():
    store = TelemetryStore()

    assert store.resize(10) is None


def test_persist_loop_flushes_after_a_publish(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    store = TelemetryStore()
    store.configure_history("Game", redis_url="redis://example", key_prefix="prefix")
    client = FakeRedisClient.instances[-1]

    store.publish_episode({"won": True})

    for _ in range(20):
        if "prefix:Game" in client.store:
            break
        time.sleep(0.05)

    assert "prefix:Game" in client.store


def test_module_level_helpers_delegate_to_the_shared_store(monkeypatch):
    fake_store = TelemetryStore()
    monkeypatch.setattr(telemetry_module, "_store", fake_store)

    telemetry_module.publish_episode(won=True)
    telemetry_module.publish_metadata("run", {"a": 1})
    telemetry_module.clear_metadata("run", clear_history=True)

    assert telemetry_module.get_store() is fake_store
    assert telemetry_module.get_store().snapshot()["summary"]["episodes"] == 0


def test_module_level_configure_history_delegates_to_the_shared_store(monkeypatch):
    FakeRedisClient.instances.clear()
    monkeypatch.setattr(telemetry_module, "Redis", FakeRedisClient)
    fake_store = TelemetryStore()
    monkeypatch.setattr(telemetry_module, "_store", fake_store)

    telemetry_module.configure_history("Game", redis_url="redis://example", key_prefix="prefix")

    assert fake_store._history_key == "prefix:Game"
