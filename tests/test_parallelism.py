from pathlib import Path

import pytest

from datenwissenschaften import parallelism
from datenwissenschaften.parallelism import optimal_env_count


def _clear_caches():
    optimal_env_count.cache_clear()


def test_rejects_non_positive_worker_limit():
    _clear_caches()
    with pytest.raises(ValueError, match="worker_limit"):
        optimal_env_count(0)
    _clear_caches()


def test_explicit_cpu_and_memory_bounds_select_the_minimum():
    _clear_caches()
    selected = optimal_env_count(cpu_count=3, memory_limit=2 * 1024**3)
    assert selected >= 1
    _clear_caches()


def test_worker_limit_caps_the_selection():
    _clear_caches()
    selected = optimal_env_count(5, cpu_count=10_000, memory_limit=1_000 * 1024**3)
    assert selected == 5
    _clear_caches()


def test_available_cpu_count_falls_back_when_sched_getaffinity_is_missing(monkeypatch):
    monkeypatch.delattr(parallelism.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(parallelism.os, "cpu_count", lambda: 5)
    monkeypatch.setattr(parallelism, "_cpu_quota", lambda: None)

    assert parallelism._available_cpu_count() == 5


def test_available_cpu_count_uses_sched_getaffinity_when_present(monkeypatch):
    monkeypatch.setattr(parallelism.os, "sched_getaffinity", lambda pid: {0, 1, 2}, raising=False)
    monkeypatch.setattr(parallelism, "_cpu_quota", lambda: None)

    assert parallelism._available_cpu_count() == 3


def test_available_cpu_count_is_capped_by_cgroup_quota(monkeypatch):
    monkeypatch.setattr(parallelism.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    monkeypatch.setattr(parallelism, "_cpu_quota", lambda: 2)

    assert parallelism._available_cpu_count() == 2


def test_cpu_quota_reads_cgroup_v2_file(monkeypatch):
    monkeypatch.setattr(parallelism.Path, "read_text", lambda self, encoding: "200000 100000")

    assert parallelism._cpu_quota() == 2


def test_cpu_quota_treats_unlimited_v2_quota_as_no_limit(monkeypatch):
    monkeypatch.setattr(parallelism.Path, "read_text", lambda self, encoding: "max 100000")

    assert parallelism._cpu_quota() is None


def test_cpu_quota_falls_back_to_cgroup_v1_when_v2_file_is_missing(monkeypatch):
    def fake_read_text(self, *, encoding):
        if self.name == "cpu.max":
            raise OSError("missing")
        if self.name == "cpu.cfs_quota_us":
            return "400000"
        if self.name == "cpu.cfs_period_us":
            return "100000"
        raise OSError("missing")

    monkeypatch.setattr(parallelism.Path, "read_text", fake_read_text)

    assert parallelism._cpu_quota() == 4


def test_cpu_quota_is_none_when_nothing_is_readable(monkeypatch):
    def fake_read_text(self, *, encoding):
        raise OSError("missing")

    monkeypatch.setattr(parallelism.Path, "read_text", fake_read_text)
    monkeypatch.setattr(parallelism, "_read_integer", lambda path: None)

    assert parallelism._cpu_quota() is None


def test_memory_limit_prefers_the_smallest_of_host_and_cgroup_limits(monkeypatch):
    monkeypatch.setattr(parallelism, "_host_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        parallelism,
        "_read_integer",
        lambda path: 2 * 1024**3 if path.name == "memory.max" else None,
    )

    assert parallelism._memory_limit() == 2 * 1024**3


def test_memory_limit_falls_back_to_cgroup_v1_when_v2_is_absent(monkeypatch):
    monkeypatch.setattr(parallelism, "_host_memory", lambda: 8 * 1024**3)
    monkeypatch.setattr(
        parallelism,
        "_read_integer",
        lambda path: 3 * 1024**3 if path.name == "memory.limit_in_bytes" else None,
    )

    assert parallelism._memory_limit() == 3 * 1024**3


def test_host_memory_parses_proc_meminfo(monkeypatch):
    monkeypatch.setattr(
        parallelism.Path,
        "read_text",
        lambda self, encoding: "MemTotal:       16777216 kB\nMemFree: 1 kB\n",
    )

    assert parallelism._host_memory() == 16777216 * 1024


def test_host_memory_defaults_when_proc_meminfo_is_unreadable(monkeypatch):
    def fake_read_text(self, *, encoding):
        raise OSError("missing")

    monkeypatch.setattr(parallelism.Path, "read_text", fake_read_text)

    assert parallelism._host_memory() == 4 * 1024**3


def test_read_integer_returns_none_for_max_sentinel(tmp_path: Path):
    path = tmp_path / "limit"
    path.write_text("max", encoding="utf-8")

    assert parallelism._read_integer(path) is None


def test_read_integer_treats_huge_values_as_unlimited(tmp_path: Path):
    path = tmp_path / "limit"
    path.write_text(str(1 << 61), encoding="utf-8")

    assert parallelism._read_integer(path) is None


def test_read_integer_returns_none_when_file_is_missing(tmp_path: Path):
    assert parallelism._read_integer(tmp_path / "missing") is None
