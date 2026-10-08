import string
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.levels import Levels, level_map
from datenwissenschaften.environment.observer import Observation, Observer
from datenwissenschaften.environment.recording import active_movie_path, ensure_movie_directory, restore_emulator_state
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)

MAX_TEXT_LENGTH = 8_192
ACTION_TABLE_DIMENSIONS = 3
SPEEDRUN_FRAME_COST = 0.005
MAX_STATE_SECONDS = 180.0
TEXT_SPACE = gym.spaces.Text(max_length=MAX_TEXT_LENGTH, charset=string.printable)


def require_action_table(action_table: np.ndarray, action_descriptions: dict[str, str]) -> None:
    if action_table.ndim != ACTION_TABLE_DIMENSIONS:
        raise ValueError("action_table must have the shape (actions, frames, buttons).")
    if len(action_descriptions) != len(action_table):
        raise ValueError("Every action_table row needs exactly one action description.")


def state_class(state_classes: tuple[type[State[T]], ...], state_name: str) -> type[State[T]]:
    for state_cls in state_classes:
        if state_cls.__name__ == state_name:
            return state_cls
    raise ValueError(f"Unknown state: {state_name}")


def step_details(ram: RamInfo, won: bool, transition: tuple[str, str] | None) -> dict[str, Any]:
    return {
        "won": won,
        "state_transition": transition,
        "ram": ram.describe(),
        "memory": ram.memory(),
        "location": ram.location(),
    }


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_table: np.ndarray
    action_descriptions: dict[str, str]
    levels: Levels

    def __init__(self, env: gym.Env, curriculum: CurriculumRun, landmarks: Landmarks, initial_savestate: str) -> None:
        super().__init__(env)
        require_action_table(self.action_table, self.action_descriptions)
        level_map(self.levels, self.state_classes)
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.observation_space = gym.spaces.Dict({"state": TEXT_SPACE, "question": TEXT_SPACE})
        self.state_machine = StateMachine[T](self.start_state_cls, landmarks)
        self.curriculum = curriculum
        self.initial_savestate = initial_savestate
        self._episode_info: dict[str, Any] = {}
        self.frames: list[np.ndarray] = []
        self.state_frames = 0
        self.max_state_frames = round(MAX_STATE_SECONDS * env.unwrapped.em.get_screen_rate())
        self.speedrun = False
        self.observer = Observer[T](env.unwrapped.get_ram, self.state_machine, tuple(self.action_descriptions.values()))

    def reset(self, **kwargs: Any) -> tuple[Observation, dict[str, Any]]:
        emulator = self.env.unwrapped
        ensure_movie_directory(emulator)
        frame, _ = self.env.reset(**kwargs)
        self.frames = [frame]
        state = self.curriculum.begin_episode()
        return self.start_from(state, None if state is None else self.curriculum.checkpoint(state), frame)

    def start_from(
        self, checkpoint_state: str | None, emulator_state: bytes | None, frame: np.ndarray
    ) -> tuple[Observation, dict[str, Any]]:
        emulator = self.env.unwrapped
        if checkpoint_state is not None and emulator_state is not None:
            restore_emulator_state(emulator, emulator_state)
            frame, *_ = self.env.step(np.zeros_like(self.action_table[0][0]))
            self.frames = [frame]
        ram = self.read_ram()
        state_classes = (self.start_state_cls, *self.state_classes)
        state_type = None if checkpoint_state is None else state_class(state_classes, checkpoint_state)
        self.state_machine.reset(ram, frame, state_type)
        self.state_frames = 0
        self._episode_info = {
            "started_from_initial_savestate": checkpoint_state is None,
            "episode_start_state": checkpoint_state or self.initial_savestate,
            "episode_start_score": self.curriculum.episode_score,
            "episode_bk2_path": active_movie_path(emulator) if emulator.movie_path else None,
        }
        observation = self.observer.observation(ram)
        return observation, {**self._step_view(), **self._episode_info}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        reward, terminated, truncated = 0.0, False, False
        transition: tuple[str, str] | None = None
        succeeded, mastered = False, False
        self.frames = []
        for buttons in self.current_action_table()[action]:
            self.curriculum.count_step()
            self.state_frames += 1
            frame, _, _, env_truncated, _ = self.env.step(buttons)
            self.frames.append(frame)
            ram = self.read_ram()
            state_reward, state_terminated, state_truncated = self.state_machine.step(ram, frame)
            reward += state_reward - (SPEEDRUN_FRAME_COST if self.speedrun else 0.0)
            terminated = state_terminated
            truncated = env_truncated or state_truncated
            transition = self.state_machine.last_transition
            if transition is not None:
                self.state_frames = 0
                emulator_state = bytes(self.env.unwrapped.em.get_state())
                succeeded, mastered = self.curriculum.transition(*transition, emulator_state, reward)
                terminated = terminated or succeeded
            truncated = truncated or self.state_frames >= self.max_state_frames
            if transition is not None or terminated or truncated:
                break

        won = self.state_machine.current_state._won()
        outcome = self.curriculum.finish_step(won, terminated or truncated, reward, transition, (succeeded, mastered))
        terminated = terminated or won
        observation = self.observer.observation(ram)
        info = {**self._step_view(), **outcome, **step_details(ram, won, transition), **self._episode_info}
        return observation, reward, terminated, truncated, info

    def current_action_table(self) -> np.ndarray:
        return self.action_table

    def _step_view(self) -> dict[str, Any]:
        return {
            "state": self.state_machine.state_name,
            "detections": self.state_machine.current_state.detections(),
            "advice": self.observer.advised_action(),
        }

    def read_ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.unwrapped.get_ram())
