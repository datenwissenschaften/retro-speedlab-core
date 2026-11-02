import random
from pathlib import Path

import numpy as np

from datenwissenschaften.route.route_map import changed_at

POLICY_FILE = "policy.npz"
BYTE_SCALE = 255.0

WEIGHT = "weight"
BIAS = "bias"
VALUE_WEIGHT = "value_weight"
VALUE_BIAS = "value_bias"

Layers = list[tuple[np.ndarray, np.ndarray]]
Weights = dict[str, np.ndarray]


def layers_of(weights: Weights) -> Layers:
    count = sum(name.startswith(WEIGHT) for name in weights)
    return [(weights[f"{WEIGHT}{index}"], weights[f"{BIAS}{index}"]) for index in range(count)]


def load_weights(path: Path) -> Weights:
    with np.load(path) as arrays:
        return {name: arrays[name] for name in arrays.files}


def observation(memory: np.ndarray) -> np.ndarray:
    return memory.astype(np.float32) / BYTE_SCALE


def hidden_of(layers: Layers, memory: np.ndarray) -> np.ndarray:
    hidden = observation(memory)
    for weight, bias in layers[:-1]:
        hidden = np.tanh(weight @ hidden + bias)
    return hidden


def log_probabilities(layers: Layers, hidden: np.ndarray, actions: int) -> np.ndarray:
    logits = (layers[-1][0] @ hidden + layers[-1][1])[:actions]
    shifted = logits - logits.max()
    return shifted - np.log(np.exp(shifted).sum())


class PolicyPrior:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.layers: Layers = []
        self.loaded: int | None = None

    def current(self) -> Layers:
        changed = changed_at(self.path)
        if changed != self.loaded:
            self.layers = [] if changed is None else layers_of(load_weights(self.path))
            self.loaded = changed
        return self.layers

    def probabilities(self, memory: np.ndarray, actions: int) -> np.ndarray | None:
        layers = self.current()
        if not layers:
            return None
        return np.exp(log_probabilities(layers, hidden_of(layers, memory), actions))

    def choose(self, memory: np.ndarray, actions: int, rng: random.Random) -> int | None:
        probabilities = self.probabilities(memory, actions)
        if probabilities is None:
            return None
        return rng.choices(range(actions), weights=probabilities.tolist())[0]
