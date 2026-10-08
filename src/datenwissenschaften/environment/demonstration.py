from pathlib import Path

import numpy as np
import stable_retro as retro
from loguru import logger
from stable_retro import RetroEnv

from datenwissenschaften.environment.recording import recorded_buttons
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.imitation import DemonstrationStep

Demonstrations = dict[str, list[DemonstrationStep]]


def load_demonstrations(wrapper: StateMachineGymWrapper, directory: Path) -> Demonstrations:
    emulator = wrapper.env.unwrapped
    record_dir = emulator.movie_path
    demonstrations: Demonstrations = {}
    emulator.stop_record()
    try:
        for path in sorted(directory.glob("*.bk2")):
            for state, steps in replay(wrapper, movie_buttons(path, emulator)).items():
                demonstrations.setdefault(state, []).extend(steps)
    finally:
        emulator.auto_record(record_dir)
    for state, steps in demonstrations.items():
        logger.info(f"Loaded {len(steps)} demonstration decisions for {state}")
    return demonstrations


def movie_buttons(path: Path, emulator: RetroEnv) -> np.ndarray:
    movie = retro.Movie(str(path))
    if movie.get_game() != emulator.gamename:
        raise ValueError(f"{path.name} is a movie of {movie.get_game()}, not {emulator.gamename}.")
    if movie.get_state() != emulator.initial_state:
        raise ValueError(f"{path.name} does not start at power-on.")
    return recorded_buttons(movie, emulator.num_buttons)


def replay(wrapper: StateMachineGymWrapper, buttons: np.ndarray) -> Demonstrations:
    frames_per_decision = wrapper.action_table.shape[1]
    frame, _ = wrapper.env.reset()
    ram = wrapper.read_ram()
    wrapper.state_machine.reset(ram, frame, None)
    demonstrations: Demonstrations = {}
    for start in range(0, len(buttons) - frames_per_decision + 1, frames_per_decision):
        chunk = buttons[start : start + frames_per_decision]
        if chunk.any():
            observation = wrapper.observer.observation(ram)
            action = int((wrapper.current_action_table() != chunk).sum(axis=(1, 2)).argmin())
            inputs = wrapper.observer.inputs(ram)
            step = DemonstrationStep(observation["state"], observation["question"], action, inputs)
            demonstrations.setdefault(wrapper.state_machine.state_name, []).append(step)
        for frame_buttons in chunk:
            frame, *_ = wrapper.env.step(frame_buttons)
            ram = wrapper.read_ram()
            wrapper.state_machine.step(ram, frame)
    return demonstrations
