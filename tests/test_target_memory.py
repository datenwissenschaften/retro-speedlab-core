from types import SimpleNamespace

from datenwissenschaften.states import target_memory as target_memory_module
from datenwissenschaften.states.target_memory import TargetMemory


def test_reset_all_deletes_persisted_and_live_target_memories(monkeypatch):
    deleted = []
    memory = SimpleNamespace(
        _store=SimpleNamespace(delete=lambda *scope: deleted.append(scope)),
        _scope=("target-memory", "Game", "Level", "Target"),
        coordinates=(12.0, 34.0),
    )
    monkeypatch.setattr(TargetMemory, "_registry", {"Target": memory})

    TargetMemory.reset_all()

    assert deleted == [("target-memory", "Game", "Level", "Target")]
    assert memory.coordinates is None


class _FakeRedisStore:
    def __init__(self, redis_url, *, key_prefix="datenwissenschaften"):
        self.values: dict[tuple, object] = {}

    def get(self, *parts, default=None):
        return self.values.get(parts, default)

    def set(self, *parts, value):
        self.values[parts] = value

    def delete(self, *parts):
        self.values.pop(parts, None)


def _configure(monkeypatch, *, active_savestate="Level1", persisted_active_savestate=None):
    monkeypatch.setattr(TargetMemory, "_registry", {})
    store = _FakeRedisStore("redis://example")
    monkeypatch.setattr(target_memory_module, "RedisStore", lambda url, **kwargs: store)
    monkeypatch.setattr(
        target_memory_module,
        "load_config",
        lambda: SimpleNamespace(
            ui=SimpleNamespace(redis_url="redis://example"),
            training=SimpleNamespace(game_identity="Game", active_savestate=active_savestate),
        ),
    )
    if persisted_active_savestate is not None:
        store.set("active-savestate", "Game", value=persisted_active_savestate)
    return store


def test_new_target_memory_scopes_to_the_configured_savestate(monkeypatch):
    _configure(monkeypatch, active_savestate="Level1")

    memory = TargetMemory("Target", origin=(0.0, 0.0), scale=100.0)

    assert memory._current_savestate == "Level1"
    assert memory._scope == ("target-memory", "Game", "Level1", "Target")


def test_new_target_memory_prefers_the_persisted_active_savestate(monkeypatch):
    # A rotating curriculum persists the active savestate to Redis; that must
    # win over the statically configured savestate for a process that starts
    # (or restarts) mid-rotation.
    _configure(monkeypatch, active_savestate="Level1", persisted_active_savestate="Level3")

    memory = TargetMemory("Target", origin=(0.0, 0.0), scale=100.0)

    assert memory._current_savestate == "Level3"


def test_remembered_target_does_not_leak_across_savestate_rotation(monkeypatch):
    _configure(monkeypatch, active_savestate="Level1")
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.remember((10.0, 20.0))
    assert memory.coordinates == (10.0, 20.0)

    TargetMemory.set_active_savestate("Level2")

    assert memory.coordinates is None
    assert memory._scope == ("target-memory", "Game", "Level2", "Target")

    # The forgotten position for Level1 is still there if we rotate back.
    TargetMemory.set_active_savestate("Level1")
    assert memory.coordinates == (10.0, 20.0)


def test_savestate_rotation_is_a_no_op_when_the_savestate_is_unchanged(monkeypatch):
    _configure(monkeypatch, active_savestate="Level1")
    memory = TargetMemory.shared("Target", origin=(0.0, 0.0), scale=100.0)
    memory.remember((10.0, 20.0))

    TargetMemory.set_active_savestate("Level1")

    assert memory.coordinates == (10.0, 20.0)
