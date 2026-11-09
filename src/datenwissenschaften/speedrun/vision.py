import subprocess
import tempfile
from pathlib import Path

import cv2
import msgspec
import numpy as np
from loguru import logger

from datenwissenschaften.speedrun.hints import Hint, Press, pressed
from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.search import Crossing, Searched, replay

VISION_MODEL = "haiku"
VISION_MODE = "plan"
VISION_SECONDS = 120
MOST_FRAMES = 600
PAUSE = "START"
PROMPT = """You play the NES game in the screenshot {image}. Read it with the Read tool.
Goal right now: {goal}
What is known about the game:
{notes}
Buttons: {buttons}. One frame is 1/60 s.
Plan the next few seconds that reach the goal or get clearly closer to it, and reply with JSON only:
a list of [frames, [buttons]] steps, at most {most} frames in total,
for example [[30, ["RIGHT"]], [12, ["RIGHT", "A"]]]."""


def screenshot(runner: SegmentRunner, folder: Path) -> Path:
    image = folder / "screen.png"
    screen = cv2.resize(runner.emulator.em.get_screen(), None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(image), cv2.cvtColor(screen, cv2.COLOR_RGB2BGR))
    return image


def asked(prompt: str, folder: Path) -> str:
    command = ["claude", "-p", prompt, "--model", VISION_MODEL, "--allowedTools", "Read", "--add-dir", str(folder)]
    return subprocess.run(command, capture_output=True, text=True, check=True, timeout=VISION_SECONDS).stdout


def playable(runner: SegmentRunner) -> list[str]:
    return [name for name in runner.emulator.buttons if name and name != PAUSE]


def unpaused(runner: SegmentRunner, frames: np.ndarray) -> np.ndarray:
    names = list(runner.emulator.buttons)
    if PAUSE in names:
        frames[:, names.index(PAUSE)] = 0
    return frames


def planned(answer: str, state: str) -> Hint:
    text = answer.strip()
    steps = msgspec.json.decode(text[text.index("[") : text.rindex("]") + 1], type=list[Press])
    return Hint(state, steps)


def vision_search(runner: SegmentRunner, start: Start, goal: str, notes: str, live: WorkerLive | None) -> Searched:
    runner.restore(start)
    if live is not None:
        live.step(runner, 0, None, 0, None)
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        prompt = PROMPT.format(
            image=screenshot(runner, folder),
            goal=goal,
            notes=notes,
            buttons=", ".join(playable(runner)),
            most=MOST_FRAMES,
        )
        try:
            answer = asked(prompt, folder)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            logger.warning(f"The vision scout got no plan for {start.state}: {error}")
            return Searched([], None)
    try:
        frames = unpaused(runner, pressed(runner, planned(answer, start.state))[:MOST_FRAMES])
    except (ValueError, msgspec.ValidationError) as error:
        logger.warning(f"The vision scout's plan for {start.state} could not be read: {error}")
        return Searched([], None)
    crossing: Crossing | None = replay(runner, start, np.asarray(frames, dtype=np.int8)) if len(frames) else None
    if live is not None:
        live.step(runner, 0, None if crossing is None else crossing.frames, len(frames), None)
    logger.info(f"The vision scout's {len(frames)}-frame plan {'leaves' if crossing else 'stays in'} {start.state}")
    return Searched([] if crossing is None else [crossing], None)
