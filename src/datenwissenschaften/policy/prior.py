import random
from pathlib import Path

import numpy as np

from datenwissenschaften.route.route_map import changed_at

POLICY_FILE = "policy.npz"
BYTE_SCALE = 255.0

Layers = list[tuple[np.ndarray, np.ndarray]]


def load_layers(path: Path) -> Layers:
    with np.load(path) as arrays:
        return [(arrays[f"weight{index}"], arrays[f"bias{index}"]) for index in range(len(arrays.files) // 2)]


def observation(memory: np.ndarray) -> np.ndarray:
    return memory.astype(np.float32) / BYTE_SCALE


class PolicyPrior:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.layers: Layers = []
        self.loaded: int | None = None

    def current(self) -> Layers:
        changed = changed_at(self.path)
        if changed != self.loaded:
            self.layers = [] if changed is None else load_layers(self.path)
            self.loaded = changed
        return self.layers

    def probabilities(self, memory: np.ndarray, actions: int) -> np.ndarray | None:
        layers = self.current()
        if not layers:
            return None
        hidden = observation(memory)
        for weight, bias in layers[:-1]:
            hidden = np.tanh(weight @ hidden + bias)
        logits = (layers[-1][0] @ hidden + layers[-1][1])[:actions]
        exponents = np.exp(logits - logits.max())
        return exponents / exponents.sum()

    def choose(self, memory: np.ndarray, actions: int, rng: random.Random) -> int | None:
        probabilities = self.probabilities(memory, actions)
        if probabilities is None:
            return None
        return rng.choices(range(actions), weights=probabilities.tolist())[0]
