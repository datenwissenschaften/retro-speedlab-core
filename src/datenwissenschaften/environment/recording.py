from pathlib import Path

import numpy as np
import stable_retro as retro
from stable_retro import RetroEnv


def active_movie_path(emulator: RetroEnv) -> str:
    if not emulator.movie_path or emulator.movie_id < 1:
        raise RuntimeError("Stable Retro is not recording; every Laya episode must be recorded.")
    state_name = Path(str(emulator.statename)).stem
    return str(Path(emulator.movie_path) / f"{emulator.gamename}-{state_name}-{emulator.movie_id - 1:06d}.bk2")


def ensure_movie_directory(emulator: RetroEnv) -> None:
    if emulator.movie_path:
        Path(emulator.movie_path).mkdir(parents=True, exist_ok=True)


def restore_emulator_state(emulator: RetroEnv, emulator_state: bytes) -> None:
    emulator.em.set_state(emulator_state)
    if emulator.movie is not None:
        emulator.movie.set_state(emulator_state)
    emulator.data.reset()
    emulator.data.update_ram()


def recorded_buttons(movie: retro.Movie, buttons: int) -> np.ndarray:
    frames = []
    while movie.step():
        frames.append([movie.get_key(button, 0) for button in range(buttons)])
    return np.asarray(frames[1:], dtype=np.int8)
