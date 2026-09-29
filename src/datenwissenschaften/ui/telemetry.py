from __future__ import annotations

import atexit
import threading
import time
from collections.abc import Callable
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from datenwissenschaften.persistence import JsonDatabase
from datenwissenschaften.serialization import to_json_value


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _empty_summary() -> dict[str, Any]:
    return {
        "episodes": 0,
        "wins": 0,
        "full_run_episodes": 0,
        "full_run_wins": 0,
        "full_run_timed_episodes": 0,
        "full_run_duration_seconds_total": 0.0,
        "full_run_best_fitness": None,
        "latest_full_run_duration_seconds": None,
        "timed_episodes": 0,
        "duration_seconds_total": 0.0,
        "best_fitness": None,
        "latest_index": None,
        "latest_timestamp": None,
        "latest_training_state": None,
        "latest_duration_seconds": None,
        "latest_final_state": None,
        "by_state": {},
        "by_savestate": {},
    }


def _state_summary(summary: dict[str, Any], state: str) -> dict[str, Any]:
    by_state = summary.setdefault("by_state", {})
    current = by_state.get(state)
    if not isinstance(current, dict):
        current = _empty_summary()
        current.pop("by_state", None)
        by_state[state] = current
    return current


def _savestate_summary(summary: dict[str, Any], savestate: str) -> dict[str, Any]:
    by_savestate = summary.setdefault("by_savestate", {})
    current = by_savestate.get(savestate)
    if not isinstance(current, dict):
        current = _empty_summary()
        current.pop("by_state", None)
        current.pop("by_savestate", None)
        by_savestate[savestate] = current
    return current


def _summarize_episode(summary: dict[str, Any], episode: dict[str, Any]) -> None:
    _update_summary_bucket(summary, episode)
    state = episode.get("training_state")
    if isinstance(state, str) and state:
        _update_summary_bucket(_state_summary(summary, state), episode)
    savestate = episode.get("savestate")
    if isinstance(savestate, str) and savestate:
        _update_summary_bucket(_savestate_summary(summary, savestate), episode)


def _update_summary_bucket(bucket: dict[str, Any], episode: dict[str, Any]) -> None:
    bucket["episodes"] = int(bucket.get("episodes", 0)) + 1
    full_run = episode.get("started_from_initial_savestate") is True
    if full_run:
        bucket["full_run_episodes"] = int(bucket.get("full_run_episodes", 0)) + 1
    if episode.get("won") is True:
        bucket["wins"] = int(bucket.get("wins", 0)) + 1
        if full_run:
            bucket["full_run_wins"] = int(bucket.get("full_run_wins", 0)) + 1

    duration = episode.get("duration_seconds")
    if isinstance(duration, int | float) and not isinstance(duration, bool):
        bucket["timed_episodes"] = int(bucket.get("timed_episodes", 0)) + 1
        bucket["duration_seconds_total"] = float(bucket.get("duration_seconds_total", 0.0)) + float(duration)
        if full_run:
            bucket["full_run_timed_episodes"] = int(bucket.get("full_run_timed_episodes", 0)) + 1
            bucket["full_run_duration_seconds_total"] = float(
                bucket.get("full_run_duration_seconds_total", 0.0)
            ) + float(duration)
            bucket["latest_full_run_duration_seconds"] = float(duration)

    fitness = episode.get("fitness")
    if isinstance(fitness, int | float) and not isinstance(fitness, bool):
        best = bucket.get("best_fitness")
        if best is None or float(fitness) > float(best):
            bucket["best_fitness"] = float(fitness)
        if full_run:
            full_run_best = bucket.get("full_run_best_fitness")
            if full_run_best is None or float(fitness) > float(full_run_best):
                bucket["full_run_best_fitness"] = float(fitness)

    index = episode.get("index")
    if isinstance(index, int):
        bucket["latest_index"] = index
    timestamp = episode.get("timestamp")
    if isinstance(timestamp, str):
        bucket["latest_timestamp"] = timestamp
    training_state = episode.get("training_state")
    if isinstance(training_state, str):
        bucket["latest_training_state"] = training_state
    if isinstance(duration, int | float) and not isinstance(duration, bool):
        bucket["latest_duration_seconds"] = float(duration)
    final_state = episode.get("final_state")
    if isinstance(final_state, str):
        bucket["latest_final_state"] = final_state


def _coerce_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = _empty_summary()
    for key in (
        "episodes",
        "wins",
        "full_run_episodes",
        "full_run_wins",
        "timed_episodes",
        "full_run_timed_episodes",
    ):
        if isinstance(value.get(key), int) and not isinstance(value.get(key), bool) and value[key] >= 0:
            summary[key] = value[key]
    if isinstance(value.get("duration_seconds_total"), int | float) and not isinstance(
        value.get("duration_seconds_total"), bool
    ):
        summary["duration_seconds_total"] = float(value["duration_seconds_total"])
    if isinstance(value.get("full_run_duration_seconds_total"), int | float) and not isinstance(
        value.get("full_run_duration_seconds_total"), bool
    ):
        summary["full_run_duration_seconds_total"] = float(value["full_run_duration_seconds_total"])
    if isinstance(value.get("best_fitness"), int | float) and not isinstance(value.get("best_fitness"), bool):
        summary["best_fitness"] = float(value["best_fitness"])
    if isinstance(value.get("full_run_best_fitness"), int | float) and not isinstance(
        value.get("full_run_best_fitness"), bool
    ):
        summary["full_run_best_fitness"] = float(value["full_run_best_fitness"])
    if isinstance(value.get("latest_index"), int):
        summary["latest_index"] = value["latest_index"]
    if isinstance(value.get("latest_timestamp"), str):
        summary["latest_timestamp"] = value["latest_timestamp"]
    if isinstance(value.get("latest_training_state"), str):
        summary["latest_training_state"] = value["latest_training_state"]
    if isinstance(value.get("latest_duration_seconds"), int | float) and not isinstance(
        value.get("latest_duration_seconds"), bool
    ):
        summary["latest_duration_seconds"] = float(value["latest_duration_seconds"])
    if isinstance(value.get("latest_full_run_duration_seconds"), int | float) and not isinstance(
        value.get("latest_full_run_duration_seconds"), bool
    ):
        summary["latest_full_run_duration_seconds"] = float(value["latest_full_run_duration_seconds"])
    if isinstance(value.get("latest_final_state"), str):
        summary["latest_final_state"] = value["latest_final_state"]

    for group_name in ("by_state", "by_savestate"):
        grouped = value.get(group_name)
        if isinstance(grouped, dict):
            for name, grouped_summary in grouped.items():
                if not isinstance(name, str) or not name:
                    continue
                coerced = _coerce_summary(grouped_summary)
                if coerced is not None:
                    coerced.pop("by_state", None)
                    coerced.pop("by_savestate", None)
                    summary[group_name][name] = coerced
    return summary


class TelemetryStore:
    def __init__(self, max_episodes: int | None = None) -> None:
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._metadata: dict[str, Any] = {}
        self._summary: dict[str, Any] = _empty_summary()
        self._started_at = _timestamp()
        self._sequence = 0
        self._history_key: str | None = None
        self._database: JsonDatabase | None = None
        self._history_version = 0
        self._persist_event = threading.Event()
        self._writer_thread: threading.Thread | None = None

    def resize(self, max_episodes: int | None) -> None:
        return None

    def configure_history(self, scope: str, database: JsonDatabase) -> None:
        history_key = f"history:{scope}"
        self.flush()
        with self._lock:
            if self._history_key == history_key and self._database is database:
                return
            self._database = database
            self._history_key = history_key
            self._history_version += 1
            self._metadata.clear()
            self._summary = _empty_summary()
            self._sequence = 0
            self._started_at = _timestamp()
            self._load_history_locked()
            if self._writer_thread is None:
                self._writer_thread = threading.Thread(
                    target=self._persist_loop,
                    name="training-history",
                    daemon=True,
                )
                self._writer_thread.start()

    def publish_episode(self, values: dict[str, Any]) -> None:
        with self._lock:
            self._sequence += 1
            episode = {
                "index": self._sequence,
                "timestamp": _timestamp(),
                **to_json_value(values),
            }
            _summarize_episode(self._summary, episode)
            self._mark_dirty()

    def publish_metadata(self, section: str, values: dict[str, Any], *, replace: bool = False) -> None:
        with self._lock:
            serialized = to_json_value(values)
            if replace:
                self._metadata[section] = serialized
            else:
                current = self._metadata.setdefault(section, {})
                current.update(serialized)
            self._mark_dirty()

    def clear_metadata(self, section: str, *, clear_history: bool = False) -> None:
        with self._lock:
            removed = self._metadata.pop(section, None)
            if removed is not None and clear_history:
                self._summary = _empty_summary()
                self._sequence = 0
                self._started_at = _timestamp()
            if removed is not None:
                self._mark_dirty()

    def episode_count(self) -> int:
        with self._lock:
            return int(self._summary["episodes"])

    def level_episode_count(self, savestate: str) -> int:
        with self._lock:
            levels = self._summary["by_savestate"]
            return int(levels[savestate]["episodes"]) if savestate in levels else 0

    def best_fitness(self, savestate: str) -> float | None:
        with self._lock:
            levels = self._summary["by_savestate"]
            return levels[savestate]["best_fitness"] if savestate in levels else None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy(
                {
                    "status": "live",
                    "started_at": self._started_at,
                    "updated_at": _timestamp(),
                    "episodes": [],
                    "summary": self._snapshot_summary_locked(),
                    "metadata": self._metadata,
                }
            )

    def flush(self, expected_version: int | None = None) -> None:
        with self._write_lock:
            with self._lock:
                if expected_version is not None and expected_version != self._history_version:
                    return
                if self._database is None or self._history_key is None:
                    return
                database = self._database
                history_key = self._history_key
                payload = to_json_value(self._snapshot_locked())
            database.set(history_key, payload)

    def reset_for_restart(self, delete_training_artifacts: Callable[[], None]) -> None:
        with self._write_lock:
            self._persist_event.clear()
            with self._lock:
                database = self._database
                history_key = self._history_key
            delete_training_artifacts()
            if database is not None and history_key is not None:
                database.delete(history_key)
            with self._lock:
                self._metadata.clear()
                self._summary = _empty_summary()
                self._sequence = 0
                self._started_at = _timestamp()
                self._history_version += 1

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "version": 2,
            "started_at": self._started_at,
            "updated_at": _timestamp(),
            "episodes": [],
            "summary": self._snapshot_summary_locked(),
            "metadata": self._metadata,
        }

    def _snapshot_summary_locked(self) -> dict[str, Any]:
        summary = deepcopy(self._summary)
        summary["retained_episodes"] = 0
        summary["discarded_episodes"] = int(summary.get("episodes", 0))
        return summary

    def _load_history_locked(self) -> None:
        try:
            if not self._database.contains(self._history_key):
                return
            payload = self._database.get(self._history_key)
            metadata = payload.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("history fields have invalid types")
            self._metadata.update(metadata)
            summary = _coerce_summary(payload.get("summary"))
            if summary is None:
                summary = _empty_summary()
            self._summary = summary
            self._sequence = int(summary.get("latest_index") or 0)
            self._started_at = payload.get("started_at") or self._started_at
            logger.info(f"Loaded training summary from {self._history_key}")
        except (AttributeError, TypeError, ValueError) as error:
            logger.warning(f"Ignoring unreadable training UI history in {self._history_key}: {error}")

    def _mark_dirty(self) -> None:
        if self._history_key is not None:
            self._persist_event.set()

    def _persist_loop(self) -> None:
        while True:
            self._persist_event.wait()
            with self._lock:
                version = self._history_version
            time.sleep(0.25)
            with self._lock:
                if version == self._history_version:
                    self._persist_event.clear()
            self.flush(expected_version=version)


_store = TelemetryStore()


def get_store() -> TelemetryStore:
    return _store


def configure_history(scope: str, database: JsonDatabase) -> None:
    _store.configure_history(scope, database)


def clear_metadata(section: str, *, clear_history: bool = False) -> None:
    _store.clear_metadata(section, clear_history=clear_history)


def episode_count() -> int:
    return _store.episode_count()


def level_episode_count(savestate: str) -> int:
    return _store.level_episode_count(savestate)


def best_fitness(savestate: str) -> float | None:
    return _store.best_fitness(savestate)


def publish_episode(**values: Any) -> None:
    _store.publish_episode(values)


def publish_metadata(section: str, values: dict[str, Any], *, replace: bool = False) -> None:
    _store.publish_metadata(section, values, replace=replace)


atexit.register(_store.flush)
