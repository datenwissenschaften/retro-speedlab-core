from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from loguru import logger

from datenwissenschaften import gym as gym_module
from datenwissenschaften.gym import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo, ram
from datenwissenschaften.states.state import State
from datenwissenschaften.states.target_memory import TargetMemory


@dataclass
class FakeRam(RamInfo):
    counter: int = ram(0)


class FakeEmulator:
    def __init__(self) -> None:
        self.state = b"initial-emulator-state"

    def get_state(self):
        return self.state

    def set_state(self, state: bytes) -> None:
        self.state = state


class FakeData:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.update_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def update_ram(self) -> None:
        self.update_calls += 1


class FakeRetroEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, *, obs_shape: tuple[int, int, int] = (64, 64, 3)) -> None:
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
        self.em = FakeEmulator()
        self.data = FakeData()
        self.movie = None
        self.movie_path: str | None = None
        self.movie_id: int | None = None
        self.gamename: str | None = None
        self.statename: str | None = None
        self._obs_shape = obs_shape
        self.ram_value = 0
        self.step_terminated = False
        self.step_truncated = False
        self.load_state_calls: list[str] = []

    def reset(self, **kwargs):
        return self._frame(), {}

    def step(self, action):
        return self._frame(), 0.0, self.step_terminated, self.step_truncated, {}

    def get_screen(self, apply_rotation: bool = True):
        return self._frame()

    def get_ram(self):
        return [self.ram_value] * 4

    def load_state(self, savestate: str) -> None:
        self.load_state_calls.append(savestate)

    def _frame(self) -> np.ndarray:
        return np.zeros(self._obs_shape, dtype=np.uint8)


class StartState(State[FakeRam]):
    def _next(self):
        if self.ram.counter >= 5:
            return WinState
        return None

    def _terminated(self):
        return self.ram.counter == -1


class WinState(State[FakeRam]):
    def _won(self):
        return True


class AuxState(State[FakeRam]):
    def auxiliary_features(self, ram: FakeRam | None = None) -> list[float]:
        if ram is not None and ram.counter > 50:
            return [0.0, 0.0]
        return [0.0]


def _fake_config(tmp_path: Path, *, active_savestate: str | None = "Level1") -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        training=SimpleNamespace(active_savestate=active_savestate, game_identity="TestGame"),
        paths=SimpleNamespace(cache_dir=tmp_path / "cache"),
    )


def _patch_gym_dependencies(monkeypatch, tmp_path: Path, *, active_savestate: str | None = "Level1") -> None:
    monkeypatch.setattr(
        gym_module,
        "load_config",
        lambda config_path: _fake_config(tmp_path, active_savestate=active_savestate),
    )
    monkeypatch.setattr(gym_module, "setup_logging", lambda level: None)
    monkeypatch.setattr(gym_module, "publish_metadata", lambda section, values, **kwargs: None)
    monkeypatch.setattr(TargetMemory, "_registry", {})


def _wrapper_class(
    *,
    start_state_cls=StartState,
    training_state_classes: tuple = (),
    grayscale: bool = False,
    terminate_on_transition: bool = False,
    transition_bonus: float = 0.0,
    action_repeat: int = 1,
):
    return type(
        "TestStateMachineGymWrapper",
        (StateMachineGymWrapper,),
        {
            "start_state_cls": start_state_cls,
            "ram_info_cls": FakeRam,
            "training_state_classes": training_state_classes,
            "grayscale": grayscale,
            "terminate_on_transition": terminate_on_transition,
            "transition_bonus": transition_bonus,
            "action_repeat": action_repeat,
        },
    )


def _build_wrapper(monkeypatch, tmp_path: Path, *, active_savestate: str | None = "Level1", **class_kwargs):
    _patch_gym_dependencies(monkeypatch, tmp_path, active_savestate=active_savestate)
    wrapper_cls = _wrapper_class(**class_kwargs)
    env = FakeRetroEnv()
    wrapper = wrapper_cls(env, obs_size=(32, 32))
    return wrapper, env


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_builds_the_observation_and_action_spaces(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper.observation_space["visual"].shape == (3, 32, 32)
    assert wrapper.observation_space["ram"].shape == (1,)
    assert "auxiliary" not in wrapper.observation_space.spaces
    assert wrapper.action_space is env.action_space


def test_init_with_an_action_table_uses_a_discrete_action_space(monkeypatch, tmp_path: Path):
    action_table = np.array([[0, 1], [1, 0]])
    _patch_gym_dependencies(monkeypatch, tmp_path)
    wrapper_cls = _wrapper_class()
    wrapper = wrapper_cls(FakeRetroEnv(), obs_size=(32, 32), action_table=action_table)

    assert isinstance(wrapper.action_space, gym.spaces.Discrete)
    assert wrapper.action_space.n == 2


def test_init_rejects_an_empty_action_table(monkeypatch, tmp_path: Path):
    _patch_gym_dependencies(monkeypatch, tmp_path)
    wrapper_cls = _wrapper_class()

    with pytest.raises(ValueError, match="must not be empty"):
        wrapper_cls(FakeRetroEnv(), obs_size=(32, 32), action_table=np.array([]))


def test_init_computes_auxiliary_feature_count_from_configured_states(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=AuxState,
        training_state_classes=(AuxState,),
    )

    assert wrapper.auxiliary_feature_count == 1
    assert wrapper.observation_space["auxiliary"].shape == (1,)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_starts_from_the_configured_initial_savestate(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)

    observation, info = wrapper.reset()

    assert info == {}
    assert set(observation) == {"visual", "ram"}
    assert wrapper._started_from_initial_savestate is True
    assert wrapper.episode_start_state() == "Level1"
    assert env.data.reset_calls == 0


def test_reset_creates_a_missing_movie_directory(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    movies_dir = tmp_path / "movies"
    env.movie_path = str(movies_dir)

    wrapper.reset()

    assert movies_dir.is_dir()


def test_reset_restores_an_automatic_checkpoint_when_available(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
    )
    for _ in range(64):
        wrapper.curriculum.record_success("StartState", 5)
    wrapper.curriculum.save_checkpoint("WinState", b"checkpoint-bytes")
    env.movie = SimpleNamespace(set_state=lambda state: setattr(env, "movie_state", state))

    observation, _info = wrapper.reset()

    assert env.em.state == b"checkpoint-bytes"
    assert env.movie_state == b"checkpoint-bytes"
    assert env.data.reset_calls == 1
    assert env.data.update_calls == 1
    assert wrapper._started_from_initial_savestate is False
    assert wrapper.episode_start_state() == "WinState"
    assert "visual" in observation


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_with_no_transition_returns_zero_reward_and_metadata(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    wrapper.reset()

    observation, reward, terminated, truncated, info = wrapper.step(0)

    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info["state"] == "StartState"
    assert info["state_transition"] is None
    assert info["curriculum_state"] == "StartState"
    assert "visual" in observation


def test_step_records_a_transition_win_and_curriculum_success(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
    )
    wrapper.reset()
    env.ram_value = 5

    _observation, reward, terminated, _truncated, info = wrapper.step(0)

    assert info["state_transition"] == ("StartState", "WinState")
    assert info["won"] is True
    assert info["curriculum_succeeded"] is True
    assert terminated is True
    assert reward == 0.0
    assert wrapper.curriculum.wins("StartState") == 1
    assert wrapper.curriculum.has_checkpoint("WinState")


def test_step_applies_the_transition_bonus(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
        transition_bonus=2.5,
    )
    wrapper.reset()
    env.ram_value = 5

    _observation, reward, _terminated, _truncated, _info = wrapper.step(0)

    assert reward == 2.5


def test_step_terminates_immediately_when_terminate_on_transition_is_set(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
        terminate_on_transition=True,
    )
    wrapper.reset()
    env.ram_value = 5

    *_rest, terminated, _truncated, _info = wrapper.step(0)

    assert terminated is True


def test_step_records_a_failure_when_the_environment_terminates_without_a_win(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    wrapper.reset()
    env.step_terminated = True

    _observation, _reward, terminated, _truncated, info = wrapper.step(0)

    assert terminated is True
    assert info["won"] is False
    assert info["curriculum_succeeded"] is False


def test_step_logs_when_a_stagnant_checkpoint_is_deleted(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    wrapper.reset()
    monkeypatch.setattr(wrapper.curriculum, "record_failure", lambda *args, **kwargs: True)
    env.step_terminated = True

    messages: list[str] = []
    handler_id = logger.add(lambda message: messages.append(str(message)), format="{message}")
    try:
        wrapper.step(0)
    finally:
        logger.remove(handler_id)

    assert any("Deleted score-stagnant automatic checkpoint" in message for message in messages)


def test_handle_curriculum_transition_is_a_noop_once_fully_mastered(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
    )
    for state_name in ("StartState", "WinState"):
        for _ in range(64):
            wrapper.curriculum.record_success(state_name, 5)
    wrapper.reset()
    assert wrapper._curriculum_outcome_recorded is True
    env.ram_value = 5

    _observation, _reward, terminated, _truncated, info = wrapper.step(0)

    assert terminated is True
    assert info["curriculum_succeeded"] is False
    assert info["curriculum_mastered"] is False


def test_step_raises_when_the_action_repeat_loop_never_produces_an_observation(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path, action_repeat=0)
    wrapper.reset()

    with pytest.raises(RuntimeError, match="No observation produced"):
        wrapper.step(0)


def test_step_raises_when_auxiliary_features_exceed_the_configured_maximum(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=AuxState,
        training_state_classes=(AuxState,),
    )
    wrapper.reset()
    env.ram_value = 51

    with pytest.raises(RuntimeError, match="produced 2 auxiliary features"):
        wrapper.step(0)


def test_step_pads_auxiliary_features_up_to_the_configured_maximum(monkeypatch, tmp_path: Path):
    class LowAux(State[FakeRam]):
        def auxiliary_features(self, ram: FakeRam | None = None) -> list[float]:
            return [0.5]

    class HighAux(State[FakeRam]):
        def auxiliary_features(self, ram: FakeRam | None = None) -> list[float]:
            return [0.5, 0.75]

    wrapper, _env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=LowAux,
        training_state_classes=(LowAux, HighAux),
    )
    wrapper.reset()

    observation, *_rest = wrapper.step(0)

    assert observation["auxiliary"].tolist() == pytest.approx([0.5, 0.0])


# ---------------------------------------------------------------------------
# translate_action
# ---------------------------------------------------------------------------


def test_translate_action_passes_through_without_an_action_table(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper.translate_action(3) == 3


def test_translate_action_maps_a_valid_index(monkeypatch, tmp_path: Path):
    action_table = np.array([[0, 1], [1, 0]])
    _patch_gym_dependencies(monkeypatch, tmp_path)
    wrapper = _wrapper_class()(FakeRetroEnv(), obs_size=(32, 32), action_table=action_table)

    assert wrapper.translate_action(1).tolist() == [1, 0]


def test_translate_action_rejects_a_non_integer_action(monkeypatch, tmp_path: Path):
    action_table = np.array([[0, 1], [1, 0]])
    _patch_gym_dependencies(monkeypatch, tmp_path)
    wrapper = _wrapper_class()(FakeRetroEnv(), obs_size=(32, 32), action_table=action_table)

    with pytest.raises(TypeError):
        wrapper.translate_action(1.5)


def test_translate_action_rejects_an_out_of_range_action(monkeypatch, tmp_path: Path):
    action_table = np.array([[0, 1], [1, 0]])
    _patch_gym_dependencies(monkeypatch, tmp_path)
    wrapper = _wrapper_class()(FakeRetroEnv(), obs_size=(32, 32), action_table=action_table)

    with pytest.raises(ValueError, match="outside"):
        wrapper.translate_action(99)


# ---------------------------------------------------------------------------
# grayscale / observation processing
# ---------------------------------------------------------------------------


def test_grayscale_observation_has_a_single_channel(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path, grayscale=True)

    observation, _info = wrapper.reset()

    assert observation["visual"].shape == (1, 32, 32)


def test_process_observation_falls_back_to_nearest_neighbour_resize_without_cv2(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)
    monkeypatch.setattr(gym_module, "cv2", None)

    observation, _info = wrapper.reset()

    assert observation["visual"].shape == (3, 32, 32)


def test_grayscale_process_observation_falls_back_without_cv2(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path, grayscale=True)
    monkeypatch.setattr(gym_module, "cv2", None)

    observation, _info = wrapper.reset()

    assert observation["visual"].shape == (1, 32, 32)


# ---------------------------------------------------------------------------
# misc accessors
# ---------------------------------------------------------------------------


def test_features_and_policy_input(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)
    wrapper.reset()

    features = wrapper.features()
    values, state_name = wrapper.policy_input()

    assert features == values.tolist()
    assert state_name == "StartState"


def test_state_name_and_num_actions(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper.state_name() == "StartState"
    assert wrapper.num_actions() == env.action_space.n


def test_set_terminate_on_transition_and_transition_bonus(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper.set_terminate_on_transition(True) is True
    assert wrapper.state_machine.terminate_on_transition is True
    assert wrapper.set_transition_bonus(1.5) == 1.5
    assert wrapper.state_machine.transition_bonus == 1.5


def test_training_state_names_defaults_to_the_start_state(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper.training_state_names() == ["StartState"]


def test_curriculum_progress_reports_every_configured_state(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(
        monkeypatch,
        tmp_path,
        training_state_classes=(StartState, WinState),
    )

    progress = wrapper.curriculum_progress()

    assert set(progress) == {"StartState", "WinState"}
    assert progress["StartState"]["active"] is True


def test_reset_training_memory_clears_target_memory_and_curriculum(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    deleted = []
    memory = SimpleNamespace(
        _store=SimpleNamespace(delete=lambda *scope: deleted.append(scope)),
        _scope=("target-memory", "TestGame", "Level1", "Target"),
        coordinates=(1.0, 2.0),
    )
    monkeypatch.setattr(TargetMemory, "_registry", {"Target": memory})
    old_curriculum = wrapper.curriculum
    env.movie_path = str(tmp_path / "movies")

    wrapper.reset_training_memory()

    assert deleted == [("target-memory", "TestGame", "Level1", "Target")]
    assert memory.coordinates is None
    assert wrapper.curriculum is not old_curriculum
    assert Path(env.movie_path).is_dir()


# ---------------------------------------------------------------------------
# set_initial_savestate / recording-path bookkeeping
# ---------------------------------------------------------------------------


def test_set_initial_savestate_loads_state_and_rescopes_target_memory(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    rescoped = []
    monkeypatch.setattr(
        TargetMemory,
        "set_active_savestate",
        classmethod(lambda cls, savestate: rescoped.append(savestate)),
    )
    old_curriculum = wrapper.curriculum

    result = wrapper.set_initial_savestate("Level2")

    assert result == "Level2"
    assert env.load_state_calls == ["Level2"]
    assert wrapper.initial_savestate == "Level2"
    assert wrapper.curriculum is not old_curriculum
    assert rescoped == ["Level2"]


def test_set_initial_savestate_moves_the_recording_directory(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    old_movie_dir = tmp_path / "records" / "Level1" / "0"
    old_movie_dir.mkdir(parents=True)
    env.movie_path = str(old_movie_dir)

    wrapper.set_initial_savestate("Level2")

    new_movie_dir = tmp_path / "records" / "Level2" / "0"
    assert env.movie_path == str(new_movie_dir)
    assert new_movie_dir.is_dir()


def test_set_initial_savestate_is_a_noop_for_recording_when_no_movie_path_is_set(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)

    wrapper.set_initial_savestate("Level2")

    assert env.movie_path is None


def test_set_initial_savestate_leaves_an_unrelated_recording_directory_untouched(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    unrelated_dir = tmp_path / "records" / "SomeOtherState" / "0"
    unrelated_dir.mkdir(parents=True)
    env.movie_path = str(unrelated_dir)

    wrapper.set_initial_savestate("Level2")

    assert env.movie_path == str(unrelated_dir)


def test_active_movie_path_returns_none_without_full_movie_metadata(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)

    assert wrapper._active_movie_path() is None


def test_resolve_state_class_rejects_an_unknown_state_name(monkeypatch, tmp_path: Path):
    wrapper, _env = _build_wrapper(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="Unknown training state"):
        wrapper._resolve_state_class("NotAState")


def test_handle_curriculum_transition_resets_episode_steps_when_reentering_the_target(
    monkeypatch,
    tmp_path: Path,
):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
    )
    for _ in range(64):
        wrapper.curriculum.record_success("StartState", 5)
    wrapper.reset()
    assert wrapper._curriculum_start_state == "WinState"
    wrapper._curriculum_episode_steps = 3
    env.ram_value = 5

    wrapper.step(0)

    assert wrapper._curriculum_episode_steps == 0


def test_curriculum_success_logs_mastery_when_the_win_target_is_reached(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(
        monkeypatch,
        tmp_path,
        start_state_cls=StartState,
        training_state_classes=(StartState, WinState),
    )
    wrapper.reset()
    for _ in range(63):
        wrapper.curriculum.record_success("StartState", 5)
    env.ram_value = 5

    _observation, _reward, _terminated, _truncated, info = wrapper.step(0)

    assert info["curriculum_mastered"] is True


def test_active_movie_path_builds_the_bk2_filename(monkeypatch, tmp_path: Path):
    wrapper, env = _build_wrapper(monkeypatch, tmp_path)
    env.movie_path = str(tmp_path / "records")
    env.movie_id = 7
    env.gamename = "Game"
    env.statename = "Level1.state"

    path = wrapper._active_movie_path()

    assert path == str(tmp_path / "records" / "Game-Level1-000006.bk2")
