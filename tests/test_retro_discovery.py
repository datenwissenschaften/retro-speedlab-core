from pathlib import Path

import gymnasium as gym
import pytest

from datenwissenschaften.retro.discovery import GameDefinitionLoader, GameRegistry


def _write_game_package(tmp_path: Path, *, package_name: str, with_wrapper: bool) -> Path:
    game_dir = tmp_path / package_name
    game_dir.mkdir()
    (game_dir / "states.py").write_text(
        'GAME_ID = "TestGame"\nDEFAULT_STATE = "Level1"\nIGNORE_STATES = ["Bonus"]\n',
        encoding="utf-8",
    )
    if with_wrapper:
        (game_dir / "__init__.py").write_text(
            "import gymnasium as gym\n\n\nclass FakeWrapper(gym.Wrapper):\n    pass\n",
            encoding="utf-8",
        )
    else:
        (game_dir / "__init__.py").write_text("", encoding="utf-8")
    return game_dir


def test_load_reads_game_id_wrapper_default_state_and_ignore_states(tmp_path: Path):
    game_dir = _write_game_package(tmp_path, package_name="pkg_full", with_wrapper=True)

    definition = GameDefinitionLoader(game_dir).load()

    assert definition.game_id == "TestGame"
    assert issubclass(definition.wrapper, gym.Wrapper)
    assert definition.default_state == "Level1"
    assert definition.ignore_states == {"Bonus"}


def test_load_rejects_missing_directory(tmp_path: Path):
    with pytest.raises(ValueError, match="does not exist"):
        GameDefinitionLoader(tmp_path / "missing").load()


def test_load_requires_game_id(tmp_path: Path):
    game_dir = tmp_path / "pkg_no_game_id"
    game_dir.mkdir()
    (game_dir / "states.py").write_text("DEFAULT_STATE = None\n", encoding="utf-8")
    (game_dir / "__init__.py").write_text(
        "import gymnasium as gym\n\n\nclass FakeWrapper(gym.Wrapper):\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="GAME_ID"):
        GameDefinitionLoader(game_dir).load()


def test_load_requires_a_wrapper_subclass(tmp_path: Path):
    game_dir = _write_game_package(tmp_path, package_name="pkg_no_wrapper", with_wrapper=False)

    with pytest.raises(ValueError, match="gymnasium.Wrapper subclass"):
        GameDefinitionLoader(game_dir).load()


def test_load_defaults_default_state_and_ignore_states_when_absent(tmp_path: Path):
    game_dir = tmp_path / "pkg_minimal"
    game_dir.mkdir()
    (game_dir / "states.py").write_text('GAME_ID = "Minimal"\n', encoding="utf-8")
    (game_dir / "__init__.py").write_text(
        "import gymnasium as gym\n\n\nclass FakeWrapper(gym.Wrapper):\n    pass\n",
        encoding="utf-8",
    )

    definition = GameDefinitionLoader(game_dir).load()

    assert definition.default_state is None
    assert definition.ignore_states == set()


def test_game_registry_from_game_dir_exposes_wrappers_default_and_ignored_states(tmp_path: Path):
    game_dir = _write_game_package(tmp_path, package_name="pkg_registry", with_wrapper=True)

    registry = GameRegistry.from_game_dir(game_dir)

    assert set(registry.wrappers) == {"TestGame"}
    assert registry.default_states == {"TestGame": "Level1"}
    assert registry.ignored_states == {"TestGame": {"Bonus"}}


def test_game_registry_default_states_is_empty_without_a_default_state(tmp_path: Path):
    game_dir = tmp_path / "pkg_no_default"
    game_dir.mkdir()
    (game_dir / "states.py").write_text('GAME_ID = "NoDefault"\n', encoding="utf-8")
    (game_dir / "__init__.py").write_text(
        "import gymnasium as gym\n\n\nclass FakeWrapper(gym.Wrapper):\n    pass\n",
        encoding="utf-8",
    )
    registry = GameRegistry.from_game_dir(game_dir)

    assert registry.default_states == {}
    assert registry.ignored_states == {"NoDefault": set()}
