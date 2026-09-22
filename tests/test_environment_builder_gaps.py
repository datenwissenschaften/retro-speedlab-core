from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften.retro import environment as environment_module
from datenwissenschaften.retro.environment import (
    EnvironmentBuilder,
    RetroEnvironmentFactory,
    RetroVecEnvBuilder,
    SavestateResolver,
    get_last_environment_wrapper,
)


def test_savestate_resolver_returns_none_and_warns_without_any_savestate():
    resolver = SavestateResolver({})

    assert resolver.resolve("TestGame", None) is None


def test_savestate_resolver_uses_the_default_state_when_none_requested(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "Level1.state"
    state_path.write_bytes(b"\x1f\x8b" + b"rest-of-file")
    monkeypatch.setattr(
        environment_module.retro.data,
        "list_states",
        lambda game: ["Level1"],
        raising=False,
    )
    monkeypatch.setattr(
        environment_module.retro.data,
        "get_file_path",
        lambda game, filename: str(state_path),
        raising=False,
    )
    resolver = SavestateResolver({"TestGame": "Level1"})

    assert resolver.resolve("TestGame", None) == "Level1"


def test_savestate_resolver_rejects_a_state_not_in_the_available_set(monkeypatch):
    monkeypatch.setattr(environment_module.retro.data, "list_states", lambda game: [], raising=False)

    resolver = SavestateResolver({})

    assert resolver.resolve("TestGame", "Level9") is None


def test_savestate_resolver_treats_a_missing_file_path_as_invalid(monkeypatch):
    monkeypatch.setattr(environment_module.retro.data, "list_states", lambda game: ["Level1"], raising=False)
    monkeypatch.setattr(environment_module.retro.data, "get_file_path", lambda game, filename: None, raising=False)

    resolver = SavestateResolver({})

    assert resolver.resolve("TestGame", "Level1") is None


def test_savestate_resolver_treats_an_unreadable_state_file_as_invalid(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(environment_module.retro.data, "list_states", lambda game: ["Level1"], raising=False)
    monkeypatch.setattr(
        environment_module.retro.data,
        "get_file_path",
        lambda game, filename: str(tmp_path / "missing.state"),
        raising=False,
    )

    resolver = SavestateResolver({})

    assert resolver.resolve("TestGame", "Level1") is None


def test_savestate_resolver_accepts_a_real_gzip_state_file(tmp_path: Path, monkeypatch):
    state_path = tmp_path / "Level1.state"
    state_path.write_bytes(b"\x1f\x8b" + b"rest-of-file")
    monkeypatch.setattr(environment_module.retro.data, "list_states", lambda game: ["Level1"], raising=False)
    monkeypatch.setattr(
        environment_module.retro.data,
        "get_file_path",
        lambda game, filename: str(state_path),
        raising=False,
    )

    resolver = SavestateResolver({})

    assert resolver.resolve("TestGame", "Level1") == "Level1"


def test_retro_environment_factory_rejects_an_unsupported_game(tmp_path: Path):
    paths = SimpleNamespace(record_dir=tmp_path)
    resolver = SavestateResolver({"TestGame": "Level1"})
    factory = RetroEnvironmentFactory(
        paths=paths,
        wrappers={},
        savestate_resolver=resolver,
        get_game=lambda: "OtherGame",
        get_savestate=lambda: "Level1",
        set_savestate=lambda value: None,
        obs_size=(96, 96),
    )

    try:
        factory.create(0)
        raised = False
    except ValueError as error:
        raised = "Unsupported game" in str(error)

    assert raised


def test_retro_environment_factory_persists_the_resolved_savestate_when_unset(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")
    paths = SimpleNamespace(record_dir=tmp_path)
    resolver = SavestateResolver({"TestGame": "Level1"})
    monkeypatch.setattr(resolver, "resolve", lambda game, requested: "Level1")
    persisted = []

    factory = RetroEnvironmentFactory(
        paths=paths,
        wrappers={"TestGame": lambda env, *, obs_size: env},
        savestate_resolver=resolver,
        get_game=lambda: "TestGame",
        get_savestate=lambda: None,
        set_savestate=lambda value: persisted.append(value),
        obs_size=(96, 96),
    )

    factory.create(3)

    assert persisted == ["Level1"]
    assert (tmp_path / "3").is_dir()


def test_environment_builder_records_the_last_wrapper_globally(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")

    def marker_wrapper(env, *, obs_size):
        return env

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        "  roms: roms\n"
        "  models: models\n"
        "  recordings: recordings\n"
        "  cache: cache\n"
        "training:\n"
        "  game: TestGame\n"
        "  savestate: Level1\n"
        "  num_envs: 1\n"
        "log_level: INFO\n"
        "upload:\n"
        "  url: https://example.test\n"
        "  api_key: null\n",
        encoding="utf-8",
    )

    EnvironmentBuilder(marker_wrapper, config_path=config_path)

    assert get_last_environment_wrapper() is marker_wrapper


class _FakeVecEnv:
    kind = "unset"

    def __init__(self, env_fns):
        self.env_fns = env_fns
        self.num_envs = len(env_fns)


class _FakeDummyVecEnv(_FakeVecEnv):
    kind = "dummy"


class _FakeSubprocVecEnv(_FakeVecEnv):
    kind = "subproc"


class _FakeWrap:
    def __init__(self, venv, *args, **kwargs):
        self.venv = venv
        self.num_envs = venv.num_envs


def _patch_vec_env_classes(monkeypatch):
    import stable_baselines3.common.vec_env as vec_env_module

    monkeypatch.setattr(vec_env_module, "DummyVecEnv", _FakeDummyVecEnv)
    monkeypatch.setattr(vec_env_module, "SubprocVecEnv", _FakeSubprocVecEnv)
    monkeypatch.setattr(vec_env_module, "VecMonitor", _FakeWrap)
    monkeypatch.setattr(vec_env_module, "VecFrameStack", _FakeWrap)


def test_environment_builder_build_uses_dummy_vec_env_for_a_single_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")
    monkeypatch.setattr(environment_module, "import_roms", lambda config_path: None)
    _patch_vec_env_classes(monkeypatch)

    def wrapper(env, *, obs_size):
        return SimpleNamespace(env=env, obs_size=obs_size)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        "  roms: roms\n"
        "  models: models\n"
        "  recordings: recordings\n"
        "  cache: cache\n"
        "training:\n"
        "  game: TestGame\n"
        "  savestate: Level1\n"
        "  num_envs: 1\n"
        "log_level: INFO\n"
        "upload:\n"
        "  url: https://example.test\n"
        "  api_key: null\n",
        encoding="utf-8",
    )

    builder = EnvironmentBuilder(wrapper, config_path=config_path)
    venv = builder.build()

    assert venv.num_envs == 1
    assert venv.venv.venv.kind == "dummy"


def test_environment_builder_build_accepts_explicit_overrides(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")
    monkeypatch.setattr(environment_module, "import_roms", lambda config_path: None)
    _patch_vec_env_classes(monkeypatch)

    def wrapper(env, *, obs_size):
        return env

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        "  roms: roms\n"
        "  models: models\n"
        "  recordings: recordings\n"
        "  cache: cache\n"
        "training:\n"
        "  game: TestGame\n"
        "  savestate: Level1\n"
        "  num_envs: 1\n"
        "log_level: INFO\n"
        "upload:\n"
        "  url: https://example.test\n"
        "  api_key: null\n",
        encoding="utf-8",
    )

    builder = EnvironmentBuilder(wrapper, config_path=config_path)
    venv = builder.build(n_envs=2, n_stack=3)

    assert builder.n_envs == 2
    assert builder.n_stack == 3
    assert venv.venv.venv.kind == "subproc"
    assert venv.venv.venv.num_envs == 2


def test_environment_builder_build_uses_subproc_vec_env_for_multiple_environments(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")
    monkeypatch.setattr(environment_module, "import_roms", lambda config_path: None)
    _patch_vec_env_classes(monkeypatch)

    def wrapper(env, *, obs_size):
        return env

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "paths:\n"
        "  roms: roms\n"
        "  models: models\n"
        "  recordings: recordings\n"
        "  cache: cache\n"
        "training:\n"
        "  game: TestGame\n"
        "  savestate: Level1\n"
        "  num_envs: 2\n"
        "log_level: INFO\n"
        "upload:\n"
        "  url: https://example.test\n"
        "  api_key: null\n",
        encoding="utf-8",
    )

    builder = EnvironmentBuilder(wrapper, config_path=config_path)
    venv = builder.build()

    assert venv.venv.venv.kind == "subproc"
    assert venv.venv.venv.num_envs == 2


def test_retro_vec_env_builder_uses_subproc_vec_env(monkeypatch, tmp_path: Path):
    _patch_vec_env_classes(monkeypatch)

    paths = SimpleNamespace(record_dir=tmp_path)
    resolver = SavestateResolver({"TestGame": "Level1"})
    monkeypatch.setattr(resolver, "resolve", lambda game, requested: "Level1")
    factory = RetroEnvironmentFactory(
        paths=paths,
        wrappers={"TestGame": lambda env, *, obs_size: env},
        savestate_resolver=resolver,
        get_game=lambda: "TestGame",
        get_savestate=lambda: "Level1",
        set_savestate=lambda value: None,
        obs_size=(96, 96),
    )

    builder = RetroVecEnvBuilder(factory)
    venv = builder.build(3, 1)

    assert venv.venv.venv.kind == "subproc"
    assert venv.num_envs == 3
