import itertools
import time
from dataclasses import dataclass

import numpy as np

from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.runner import CellKey, SegmentRunner, Start
from datenwissenschaften.speedrun.search import CHECKPOINT_GAIN, Checkpoint, Crossing, Searched, confirmed, leaves

BEAM_WIDTH = 24
HOLDS = (6, 12, 24)
CHORDS = ((), ("RIGHT",), ("LEFT",), ("A",), ("RIGHT", "A"), ("LEFT", "A"), ("B",), ("RIGHT", "B"), ("UP",), ("DOWN",))
BEAM_MODE = "beam"


@dataclass(slots=True, frozen=True)
class Node:
    start: Start
    pressed: np.ndarray
    score: float


def macros(runner: SegmentRunner) -> list[np.ndarray]:
    names = list(runner.emulator.buttons)
    idle = np.zeros_like(runner.wrapper.current_action_table()[0][0], dtype=np.int8)
    found: list[np.ndarray] = []
    for chord, hold in itertools.product(CHORDS, HOLDS):
        if any(name not in names for name in chord):
            continue
        buttons = idle.copy()
        for name in chord:
            buttons[names.index(name)] = 1
        found.append(np.repeat(buttons[np.newaxis], hold, axis=0))
    return found


def expanded(runner: SegmentRunner, origin: Start, node: Node, macro: np.ndarray) -> Node | Crossing | None:
    runner.restore(node.start)
    machine = runner.wrapper.state_machine
    score = node.score
    for frame, buttons in enumerate(macro, 1):
        alive, won, reward = runner.press(buttons)
        score += reward
        pressed = np.concatenate([node.pressed, macro[:frame]])
        if leaves(runner, origin, machine.state_name, won):
            return confirmed(runner, pressed, won)
        if not alive or machine.state_name != origin.state:
            return None
    return Node(runner.here(origin.state), np.concatenate([node.pressed, macro]), score)


def beam_search(runner: SegmentRunner, start: Start, seconds: float, live: WorkerLive | None) -> Searched:
    deadline = time.monotonic() + seconds
    moves = macros(runner)
    empty = np.zeros((0, len(moves[0][0])), dtype=np.int8)
    beam, best = [Node(start, empty, 0.0)], Node(start, empty, 0.0)
    crossings: list[Crossing] = []
    while beam and not crossings and time.monotonic() < deadline:
        children: dict[CellKey, Node] = {}
        for node, macro in itertools.product(beam, moves):
            child = expanded(runner, start, node, macro)
            if isinstance(child, Crossing):
                crossings.append(child)
            elif child is not None:
                key = runner.key(len(child.pressed))
                if key not in children or child.score > children[key].score:
                    children[key] = child
            if live is not None:
                live.step(runner, len(children), None, len(macro), None)
        beam = sorted(children.values(), key=lambda node: node.score, reverse=True)[:BEAM_WIDTH]
        best = max([best, *beam[:1]], key=lambda node: node.score)
    reached = None if best.score < CHECKPOINT_GAIN else Checkpoint(best.start, best.pressed, best.score, None)
    return Searched(crossings, reached)
