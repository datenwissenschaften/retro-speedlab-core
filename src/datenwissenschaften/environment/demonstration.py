from pathlib import Path

import numpy as np
import stable_retro as retro
from loguru import logger
from stable_retro import RetroEnv

from datenwissenschaften.environment.recording import recorded_buttons
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.imitation import DemonstrationStep

Demonstrations = dict[str, list[DemonstrationStep]]
StartPoints = dict[str, list[bytes]]
POINT_STRIDE = 4


def load_demonstrations(wrapper: StateMachineGymWrapper, directory: Path) -> tuple[Demonstrations, StartPoints]:
    emulator = wrapper.env.unwrapped
    record_dir = emulator.movie_path
    demonstrations: Demonstrations = {}
    starts: StartPoints = {}
    emulator.stop_record()
    try:
        for path in sorted(directory.glob("*.bk2")):
            steps, points = replay(wrapper, movie_buttons(path, emulator))
            for state, state_steps in steps.items():
                demonstrations.setdefault(state, []).extend(state_steps)
            starts.update(points)
    finally:
        emulator.auto_record(record_dir)
    for state, state_steps in demonstrations.items():
        logger.info(f"Loaded {len(state_steps)} demonstration decisions for {state}")
    for state, points in starts.items():
        logger.info(f"Loaded {len(points)} backplay start points for {state}")
    return demonstrations, starts


def movie_buttons(path: Path, emulator: RetroEnv) -> np.ndarray:
    movie = retro.Movie(str(path))
    if movie.get_game() != emulator.gamename:
        raise ValueError(f"{path.name} is a movie of {movie.get_game()}, not {emulator.gamename}.")
    if movie.get_state() != emulator.initial_state:
        raise ValueError(f"{path.name} does not start at power-on.")
    return recorded_buttons(movie, emulator.num_buttons)


def replay(wrapper: StateMachineGymWrapper, buttons: np.ndarray) -> tuple[Demonstrations, StartPoints]:
    frames_per_decision = wrapper.action_table.shape[1]
    order = [state_cls.__name__ for state_cls in (wrapper.start_state_cls, *wrapper.state_classes)]
    frame, _ = wrapper.env.reset()
    ram = wrapper.read_ram()
    wrapper.state_machine.reset(ram, frame, None)
    demonstrations: Demonstrations = {}
    starts: StartPoints = {}
    segment: list[bytes] = []
    decisions = 0
    for start in range(0, len(buttons) - frames_per_decision + 1, frames_per_decision):
        state = wrapper.state_machine.state_name
        if decisions % POINT_STRIDE == 0:
            segment.append(bytes(wrapper.env.unwrapped.em.get_state()))
        decisions += 1
        chunk = buttons[start : start + frames_per_decision]
        if chunk.any():
            observation = wrapper.observer.observation(ram)
            action = int((wrapper.current_action_table() != chunk).sum(axis=(1, 2)).argmin())
            inputs = wrapper.observer.inputs(ram)
            step = DemonstrationStep(observation["state"], observation["question"], action, inputs)
            demonstrations.setdefault(state, []).append(step)
        for frame_buttons in chunk:
            frame, *_ = wrapper.env.step(frame_buttons)
            ram = wrapper.read_ram()
            wrapper.state_machine.step(ram, frame)
        reached = wrapper.state_machine.state_name
        if reached != state:
            if order.index(reached) > order.index(state):
                starts[state] = segment
            segment, decisions = [], 0
    return demonstrations, starts
