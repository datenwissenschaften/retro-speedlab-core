from types import SimpleNamespace

import pytest
from redis.exceptions import RedisError

from datenwissenschaften import persistence as persistence_module
from datenwissenschaften.persistence import RedisStore


def test_constructor_pings_redis_and_stores_the_trimmed_prefix(monkeypatch):
    fake_client = SimpleNamespace(ping=lambda: None)

    class _FakeRedis:
        @staticmethod
        def from_url(redis_url, decode_responses=False):
            return fake_client

    monkeypatch.setattr(persistence_module, "Redis", _FakeRedis)

    store = RedisStore("redis://example", key_prefix="scope:")

    assert store._redis is fake_client
    assert store._prefix == "scope"


def test_constructor_wraps_a_failed_ping_in_a_runtime_error(monkeypatch):
    def _failing_ping():
        raise RedisError("boom")

    fake_client = SimpleNamespace(ping=_failing_ping)

    class _FakeRedis:
        @staticmethod
        def from_url(redis_url, decode_responses=False):
            return fake_client

    monkeypatch.setattr(persistence_module, "Redis", _FakeRedis)

    with pytest.raises(RuntimeError, match="Could not connect to Redis store"):
        RedisStore("redis://example")


def _store(redis):
    store = RedisStore.__new__(RedisStore)
    store._redis = redis
    store._prefix = "datenwissenschaften"
    return store


def test_get_returns_the_default_when_the_key_is_missing():
    store = _store(SimpleNamespace(get=lambda key: None))

    assert store.get("scope", default="fallback") == "fallback"


def test_get_decodes_a_stored_json_value():
    store = _store(SimpleNamespace(get=lambda key: b'{"a": 1}'))

    assert store.get("scope") == {"a": 1}


def test_set_encodes_the_value_as_json():
    calls = []
    store = _store(SimpleNamespace(set=lambda key, value: calls.append((key, value))))

    store.set("scope", "id", value={"a": 1})

    assert calls == [("datenwissenschaften:scope:id", '{"a":1}')]


def test_delete_removes_the_namespaced_key():
    calls = []
    store = _store(SimpleNamespace(delete=lambda key: calls.append(key)))

    store.delete("scope", "id")

    assert calls == ["datenwissenschaften:scope:id"]
