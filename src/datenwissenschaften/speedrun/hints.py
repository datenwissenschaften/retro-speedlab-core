from pathlib import Path

import msgspec
import numpy as np
from numpy.typing import NDArray

from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing, replay

HINTS_FILE = "hints.json"


class Press(msgspec.Struct, array_like=True, forbid_unknown_fields=True):
    frames: int
    buttons: list[str]


class Hint(msgspec.Struct, forbid_unknown_fields=True):
    state: str
    presses: list[Press]


def hints_path(config: Path) -> Path:
    return config.parent / HINTS_FILE


def load_hints(path: Path) -> list[Hint]:
    if not path.is_file():
        raise RuntimeError(f"Missing {path}: write [] when the lab has no hints")
    return msgspec.json.decode(path.read_bytes(), type=list[Hint])


def pressed(runner: SegmentRunner, hint: Hint) -> np.ndarray:
    names = list(runner.emulator.buttons)
    idle = np.zeros_like(runner.wrapper.current_action_table()[0][0], dtype=np.int8)
    frames: list[NDArray[np.int8]] = []
    for press in hint.presses:
        buttons = idle.copy()
        for name in press.buttons:
            if name not in names:
                raise ValueError(f"Hint for {hint.state} presses {name}, not one of {names}")
            buttons[names.index(name)] = 1
        frames.extend([buttons] * press.frames)
    return np.asarray(frames, dtype=np.int8)


def hinted(runner: SegmentRunner, candidates: list[np.ndarray], starts: list[Start]) -> tuple[int, Crossing] | None:
    state = starts[-1].state
    for frames in candidates:
        for index, start in enumerate(starts):
            if start.state == state and (crossing := replay(runner, start, frames)) is not None:
                return index, crossing
    return None
