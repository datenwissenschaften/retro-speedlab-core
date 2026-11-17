import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np

from datenwissenschaften.kernel.batch import Batch
from datenwissenschaften.kernel.expand import LEAVE, STAY, Counts, Wide
from datenwissenschaften.speedrun.live import WorkerLive
from datenwissenschaften.speedrun.runner import CellKey, SegmentRunner, Start, cell_key
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


Layer = tuple[dict[CellKey, Node], list[Crossing]]
Expand = Callable[[SegmentRunner, Start, list[Node], list[np.ndarray], WorkerLive | None], Layer]


def macros(runner: SegmentRunner) -> list[np.ndarray]:
    names = list(runner.emulator.buttons)
    idle = np.zeros_like(runner.wrapper.action_table[0][0], dtype=np.int8)
    found: list[np.ndarray] = []
    for chord, hold in itertools.product(CHORDS, HOLDS):
        if any(name not in names for name in chord):
            continue
        buttons = idle.copy()
        for name in chord:
            buttons[names.index(name)] = 1
        found.append(np.repeat(buttons[np.newaxis], hold, axis=0))
    return found


def packed(moves: list[np.ndarray]) -> tuple[Wide, Counts]:
    padded = np.zeros((len(moves), max(map(len, moves)), moves[0].shape[1]), np.int32)
    for index, macro in enumerate(moves):
        padded[index, : len(macro)] = macro
    return padded, np.array([len(macro) for macro in moves], np.int64)


def expanded(runner: SegmentRunner, origin: Start, node: Node, macro: np.ndarray) -> Node | Crossing | None:
    runner.restore(node.start)
    machine = runner.wrapper.state_machine
    score = node.score
    for frame, buttons in enumerate(macro, 1):
        alive, won, reward = runner.press(buttons)
        score += reward
        if leaves(runner, origin, machine.state_name, won):
            return confirmed(runner, np.concatenate([node.pressed, macro[:frame]]), won)
        if not alive or machine.state_name != origin.state:
            return None
    return Node(runner.here(origin.state), np.concatenate([node.pressed, macro]), score)


def kept(runner: SegmentRunner, layer: Layer, child: Node | Crossing | None) -> None:
    if isinstance(child, Crossing):
        layer[1].append(child)
    elif child is not None:
        keep(layer, child, runner.key(len(child.pressed)))


def keep(layer: Layer, child: Node, key: CellKey) -> None:
    children = layer[0]
    if key not in children or child.score > children[key].score:
        children[key] = child


def stepped(
    runner: SegmentRunner, origin: Start, beam: list[Node], moves: list[np.ndarray], live: WorkerLive | None
) -> Layer:
    layer: Layer = ({}, [])
    for node, macro in itertools.product(beam, moves):
        kept(runner, layer, expanded(runner, origin, node, macro))
        if live is not None:
            live.step(runner, len(layer[0]), None, len(macro), None)
    return layer


def batched(
    batch: Batch,
    padded: tuple[Wide, Counts],
    runner: SegmentRunner,
    origin: Start,
    beam: list[Node],
    moves: list[np.ndarray],
    live: WorkerLive | None,
) -> Layer:
    rams = np.stack([np.frombuffer(node.start.emulator_state, np.uint8) for node in beam])
    found = batch.expand(rams, *padded, runner.order.index(origin.state))
    layer: Layer = ({}, [])
    reader = runner.wrapper.ram_info_cls
    for index, (node, macro) in enumerate(itertools.product(beam, moves)):
        if found.outcomes[index] == LEAVE:
            kept(runner, layer, expanded(runner, origin, node, macro))
        elif found.outcomes[index] == STAY:
            start = Start(origin.state, found.rams[index].tobytes(), None)
            child = Node(start, np.concatenate([node.pressed, macro]), node.score + float(found.rewards[index]))
            keep(layer, child, cell_key(origin.state, reader.from_ram(found.rams[index]), len(child.pressed)))
    if live is not None:
        runner.restore(max(layer[0].values(), key=lambda child: child.score, default=beam[0]).start)
        live.step(runner, len(layer[0]), None, int(found.frames.sum()), None)
    return layer


def beam_search(
    runner: SegmentRunner, start: Start, seconds: float, live: WorkerLive | None, batch: Batch | None
) -> Searched:
    deadline = time.monotonic() + seconds
    moves = macros(runner)
    expand: Expand = stepped if batch is None else partial(batched, batch, packed(moves))
    empty = np.zeros((0, len(moves[0][0])), dtype=np.int8)
    beam, best = [Node(start, empty, 0.0)], Node(start, empty, 0.0)
    crossings: list[Crossing] = []
    while beam and not crossings and time.monotonic() < deadline:
        children, crossings = expand(runner, start, beam, moves, live)
        beam = sorted(children.values(), key=lambda node: node.score, reverse=True)[:BEAM_WIDTH]
        best = max([best, *beam[:1]], key=lambda node: node.score)
    reached = None if best.score < CHECKPOINT_GAIN else Checkpoint(best.start, best.pressed, best.score, None)
    return Searched(crossings, reached)
