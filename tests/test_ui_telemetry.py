import time

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.ui import telemetry as telemetry_module
from datenwissenschaften.ui.telemetry import TelemetryStore


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


def test_flush_is_a_noop_when_history_is_not_configured():
    store = TelemetryStore()

    store.flush()


def test_reset_for_restart_without_redis_configured_still_runs_cleanup():
    store = TelemetryStore()

    cleanup_calls = []
    store.reset_for_restart(lambda: cleanup_calls.append(True))

    assert cleanup_calls == [True]


def test_resize_is_a_noop():
    store = TelemetryStore()

    assert store.resize(10) is None


def test_module_level_helpers_delegate_to_the_shared_store(monkeypatch):
    fake_store = TelemetryStore()
    monkeypatch.setattr(telemetry_module, "_store", fake_store)

    telemetry_module.publish_episode(won=True)
    telemetry_module.publish_metadata("run", {"a": 1})
    telemetry_module.clear_metadata("run", clear_history=True)

    assert telemetry_module.get_store() is fake_store
    assert telemetry_module.get_store().snapshot()["summary"]["episodes"] == 0


def test_configure_history_loads_a_persisted_summary(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    database.set(
        "history:Game",
        {"started_at": "2024-01-01T00:00:00Z", "metadata": {"run": {"game": "Game"}}, "summary": {"episodes": 4}},
    )
    store = TelemetryStore()

    store.configure_history("Game", database)

    snapshot = store.snapshot()
    assert snapshot["metadata"]["run"] == {"game": "Game"}
    assert snapshot["summary"]["episodes"] == 4


def test_configure_history_is_idempotent_for_the_same_scope(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    store = TelemetryStore()
    store.configure_history("Game", database)
    store.publish_metadata("run", {"game": "Game"})

    store.configure_history("Game", database)

    assert store.snapshot()["metadata"]["run"] == {"game": "Game"}


def test_configure_history_ignores_malformed_history(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    database.set("history:Game", {"metadata": ["not", "a", "mapping"]})
    store = TelemetryStore()

    store.configure_history("Game", database)

    assert store.snapshot()["metadata"] == {}


def test_flush_persists_the_snapshot_to_the_json_database(tmp_path):
    path = tmp_path / "database.json"
    store = TelemetryStore()
    store.configure_history("Game", JsonDatabase(path))
    store.publish_episode({"fitness": 3.0, "won": True})

    store.flush()

    persisted = JsonDatabase(path).get("history:Game")
    assert persisted["summary"]["episodes"] == 1
    assert persisted["summary"]["wins"] == 1


def test_flush_skips_a_stale_history_version(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    store = TelemetryStore()
    store.configure_history("Game", database)

    store.flush(expected_version=-1)

    assert not database.contains("history:Game")


def test_reset_for_restart_deletes_persisted_history(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    store = TelemetryStore()
    store.configure_history("Game", database)
    store.publish_episode({"fitness": 1.0, "won": False})
    store.flush()
    cleaned = []

    store.reset_for_restart(lambda: cleaned.append(True))

    assert cleaned == [True]
    assert not database.contains("history:Game")
    assert store.snapshot()["summary"]["episodes"] == 0


def test_persist_loop_flushes_after_a_publish(tmp_path):
    database = JsonDatabase(tmp_path / "database.json")
    store = TelemetryStore()
    store.configure_history("Game", database)

    store.publish_episode({"fitness": 2.0, "won": False})
    deadline = time.monotonic() + 3
    while not database.contains("history:Game") and time.monotonic() < deadline:
        time.sleep(0.05)

    assert database.get("history:Game")["summary"]["episodes"] == 1


def test_module_level_configure_history_delegates_to_the_shared_store(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(telemetry_module._store, "configure_history", lambda scope, database: calls.append(scope))

    telemetry_module.configure_history("Game", JsonDatabase(tmp_path / "database.json"))

    assert calls == ["Game"]


def test_best_fitness_and_attempts_are_counted_per_level():
    store = TelemetryStore()
    assert store.best_fitness("Level1") is None

    store.publish_episode({"fitness": 2.0, "won": False, "savestate": "Level1"})
    store.publish_episode({"fitness": 5.0, "won": False, "savestate": "Level1"})
    store.publish_episode({"fitness": 9.0, "won": False, "savestate": "Level2"})

    assert (store.best_fitness("Level1"), store.best_fitness("Level2")) == (5.0, 9.0)
    assert (store.level_episode_count("Level1"), store.level_episode_count("Level3")) == (2, 0)
    assert store.episode_count() == 3
