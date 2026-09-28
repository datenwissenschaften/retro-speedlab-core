import json
import string
from typing import Any, Generic, TypeVar

import gymnasium as gym
import numpy as np

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.recording import active_movie_path, ensure_movie_directory, restore_emulator_state
from datenwissenschaften.ram import RamInfo
from datenwissenschaften.states.machine import StateMachine
from datenwissenschaften.states.state import State

T = TypeVar("T", bound=RamInfo)

MAX_TEXT_LENGTH = 8_192
ACTION_TABLE_DIMENSIONS = 3
TEXT_SPACE = gym.spaces.Text(max_length=MAX_TEXT_LENGTH, charset=string.printable)

Observation = dict[str, str]


class StateMachineGymWrapper(gym.Wrapper, Generic[T]):
    start_state_cls: type[State[T]]
    state_classes: tuple[type[State[T]], ...]
    ram_info_cls: type[T]
    action_table: np.ndarray
    action_descriptions: dict[str, str]

    def __init__(self, env: gym.Env, curriculum: CurriculumRun, initial_savestate: str) -> None:
        super().__init__(env)
        if self.action_table.ndim != ACTION_TABLE_DIMENSIONS:
            raise ValueError("action_table must have the shape (actions, frames, buttons).")
        if len(self.action_descriptions) != len(self.action_table):
            raise ValueError("Every action_table row needs exactly one action description.")
        self.action_space = gym.spaces.Discrete(len(self.action_table))
        self.observation_space = gym.spaces.Dict({"state": TEXT_SPACE, "question": TEXT_SPACE})
        self.state_machine = StateMachine[T](self.start_state_cls())
        self.curriculum = curriculum
        self.initial_savestate = initial_savestate
        self._episode_info: dict[str, Any] = {}
        self.frames: list[np.ndarray] = []

    def reset(self, **kwargs: Any) -> tuple[Observation, dict[str, Any]]:
        emulator = self.env.unwrapped
        ensure_movie_directory(emulator)
        frame, _ = self.env.reset(**kwargs)
        self.frames = [frame]
        checkpoint_state = self.curriculum.begin_episode()
        if checkpoint_state is not None:
            restore_emulator_state(emulator, self.curriculum.checkpoint(checkpoint_state))
            frame, *_ = self.env.step(np.zeros_like(self.action_table[0][0]))
            self.frames = [frame]
        ram = self._read_ram()
        state_type = None if checkpoint_state is None else self._state_class(checkpoint_state)
        self.state_machine.reset(ram, frame, state_type)
        self._episode_info = {
            "started_from_initial_savestate": checkpoint_state is None,
            "episode_start_state": checkpoint_state or self.initial_savestate,
            "episode_start_score": self.curriculum.episode_score,
            "episode_bk2_path": active_movie_path(emulator),
        }
        return self._observation(ram), {**self._step_view(), **self._episode_info}

    def step(self, action: int) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        reward, terminated, truncated = 0.0, False, False
        transition: tuple[str, str] | None = None
        succeeded, mastered = False, False
        self.frames = []
        for buttons in self.action_table[action]:
            self.curriculum.count_step()
            frame, _, env_terminated, env_truncated, _ = self.env.step(buttons)
            self.frames.append(frame)
            ram = self._read_ram()
            state_reward, state_terminated, state_truncated = self.state_machine.step(ram, frame)
            reward += state_reward
            terminated = env_terminated or state_terminated
            truncated = env_truncated or state_truncated
            transition = self.state_machine.last_transition
            if transition is not None:
                emulator_state = bytes(self.env.unwrapped.em.get_state())
                succeeded, mastered = self.curriculum.transition(*transition, emulator_state, reward)
                terminated = terminated or succeeded
            if transition is not None or terminated or truncated:
                break

        won = self.state_machine.current_state._won()
        outcome = self.curriculum.finish_step(won, terminated or truncated, reward, transition, (succeeded, mastered))
        terminated = terminated or won
        info = {
            **self._step_view(),
            **outcome,
            "won": won,
            "state_transition": transition,
            "ram": ram.describe(),
            **self._episode_info,
        }
        return self._observation(ram), reward, terminated, truncated, info

    def _step_view(self) -> dict[str, Any]:
        return {"state": self.state_machine.state_name, "detections": self.state_machine.current_state.detections()}

    def reset_training_memory(self) -> None:
        self.curriculum.reset_memory()

    def _observation(self, ram: T) -> Observation:
        state = {**ram.describe(), **self.state_machine.current_state.describe()}
        return {"state": json.dumps(state), "question": self.state_machine.question}

    def _read_ram(self) -> T:
        return self.ram_info_cls.from_ram(self.env.unwrapped.get_ram())

    def _state_class(self, state_name: str) -> type[State[T]]:
        for state_cls in (self.start_state_cls, *self.state_classes):
            if state_cls.__name__ == state_name:
                return state_cls
        raise ValueError(f"Unknown state: {state_name}")
