from dataclasses import dataclass

import torch

from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.probe.samples import Sample
from datenwissenschaften.states.facts import Offset

READ_BATCH = 16
HOLDOUT_SHARE = 0.2
SPLIT_SEED = 0
STANDARDIZE_EPSILON = 1e-6
MIN_OFFSET_R2 = 0.5


@dataclass(slots=True, frozen=True)
class Readability:
    state: str
    fact: str
    r2: float
    required: bool

    @property
    def passed(self) -> bool:
        return not self.required or self.r2 >= MIN_OFFSET_R2


def numeric_facts(sample: Sample) -> dict[str, tuple[float, bool]]:
    numbers: dict[str, tuple[float, bool]] = {}
    for name, value in sample.facts.items():
        if isinstance(value, Offset):
            numbers[f"{name}.right"] = (float(value.right), True)
            numbers[f"{name}.down"] = (float(value.down), True)
        elif isinstance(value, int | float) and not isinstance(value, bool):
            numbers[name] = (float(value), False)
    return numbers


def features(network: LayaNetwork, samples: list[Sample]) -> torch.Tensor:
    parts = []
    for start in range(0, len(samples), READ_BATCH):
        batch = samples[start : start + READ_BATCH]
        options, pooled = network.features([sample.text for sample in batch], [sample.question for sample in batch])
        parts.append(torch.cat([pooled, options.flatten(1)], -1).double().cpu())
    return torch.cat(parts)


def held_out_r2(inputs: torch.Tensor, target: torch.Tensor) -> float:
    order = torch.randperm(len(target), generator=torch.Generator().manual_seed(SPLIT_SEED))
    split = round(len(target) * (1 - HOLDOUT_SHARE))
    train, test = order[:split], order[split:]
    scaled = (inputs - inputs[train].mean(0)) / (inputs[train].std(0) + STANDARDIZE_EPSILON)
    design = torch.cat([scaled, torch.ones(len(scaled), 1, dtype=scaled.dtype)], -1)
    known = design[train]
    penalty = float(len(train)) * torch.eye(known.size(1), dtype=known.dtype)
    weights = torch.linalg.solve(known.T @ known + penalty, known.T @ target[train])
    error = design[test] @ weights - target[test]
    return float(1 - error.var() / target[test].var())


def readability(network: LayaNetwork, samples: list[Sample]) -> list[Readability]:
    results = []
    for state in dict.fromkeys(sample.state for sample in samples):
        chosen = [sample for sample in samples if sample.state == state]
        inputs = features(network, chosen)
        facts = [numeric_facts(sample) for sample in chosen]
        for name in dict.fromkeys(name for numbers in facts for name in numbers):
            if not all(name in numbers for numbers in facts):
                continue
            target = torch.tensor([numbers[name][0] for numbers in facts], dtype=torch.float64)
            if target.std() == 0:
                continue
            results.append(Readability(state, name, held_out_r2(inputs, target), facts[0][name][1]))
    return results
