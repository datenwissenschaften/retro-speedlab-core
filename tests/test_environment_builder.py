from pathlib import Path
from types import SimpleNamespace

import yaml

from datenwissenschaften.retro import environment as environment_module
from datenwissenschaften.retro.environment import EnvironmentBuilder, RetroEnvironmentFactory, SavestateResolver


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "roms": "roms",
                    "models": "models",
                    "recordings": "recordings",
                    "cache": "cache",
                },
                "training": {"game": "TestGame", "savestate": "Level1", "num_envs": 1},
                "log_level": "INFO",
                "upload": {"url": "https://example.test", "api_key": None},
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_environment_builder_passes_obs_size_to_the_wrapper(monkeypatch, tmp_path: Path):
    received = {}

    def fake_wrapper(env, *, obs_size):
        received["env"] = env
        received["obs_size"] = obs_size
        return env

    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")

    builder = EnvironmentBuilder(fake_wrapper, config_path=_write_config(tmp_path))
    builder.make_env(0)

    assert received == {"env": "raw-env", "obs_size": (96, 96)}


def test_environment_builder_obs_size_is_configurable(monkeypatch, tmp_path: Path):
    received = {}

    def fake_wrapper(env, *, obs_size):
        received["obs_size"] = obs_size
        return env

    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")

    builder = EnvironmentBuilder(fake_wrapper, obs_size=(64, 64), config_path=_write_config(tmp_path))
    builder.make_env(0)

    assert received["obs_size"] == (64, 64)


def test_retro_environment_factory_passes_the_same_obs_size_contract(monkeypatch, tmp_path: Path):
    received = {}

    def fake_wrapper(env, *, obs_size):
        received["obs_size"] = obs_size
        return env

    monkeypatch.setattr(environment_module.retro, "make", lambda *args, **kwargs: "raw-env")
    monkeypatch.setattr(
        environment_module.retro.data,
        "list_states",
        lambda game: ["Level1"],
        raising=False,
    )

    paths = SimpleNamespace(record_dir=tmp_path)
    resolver = SavestateResolver({"TestGame": "Level1"})
    monkeypatch.setattr(resolver, "resolve", lambda game, requested: "Level1")
    factory = RetroEnvironmentFactory(
        paths=paths,
        wrappers={"TestGame": fake_wrapper},
        savestate_resolver=resolver,
        get_game=lambda: "TestGame",
        get_savestate=lambda: "Level1",
        set_savestate=lambda value: None,
        obs_size=(96, 96),
    )

    factory.create(0)

    assert received["obs_size"] == (96, 96)
