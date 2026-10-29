from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import typer
from loguru import logger
from transformers import AutoModel

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.probe.samples import Sample, collect_samples
from datenwissenschaften.settings import load_config
from datenwissenschaften.states.facts import Offset
from datenwissenschaften.workflow.wrapper import wrapper_class

D1_CHECKPOINT = "LiquidAI/d1-omni-600M"
AXES = {
    "right": {"left": "to the left of Rattle", "here": "level with Rattle", "right": "to the right of Rattle"},
    "down": {"up": "above Rattle", "here": "level with Rattle", "down": "below Rattle"},
}

Reader = Callable[[list[str], str, dict[str, str]], list[str]]


@dataclass(slots=True, frozen=True)
class Question:
    fact: str
    axis: str
    texts: list[str]
    truths: list[str]

    @property
    def text(self) -> str:
        return f"Where is {self.fact} from Rattle?"


@dataclass(slots=True, frozen=True)
class Score:
    fact: str
    axis: str
    samples: int
    majority: float
    laya: float
    d1: float


def direction(value: int, options: tuple[str, str, str]) -> str:
    return options[1] if value == 0 else options[2] if value > 0 else options[0]


def questions(samples: list[Sample]) -> list[Question]:
    found: dict[tuple[str, str], tuple[list[str], list[str]]] = {}
    for sample in samples:
        for fact, value in sample.facts.items():
            if not isinstance(value, Offset):
                continue
            for axis, options in AXES.items():
                texts, truths = found.setdefault((fact, axis), ([], []))
                texts.append(sample.text)
                truths.append(direction({"right": value.right, "down": value.down}[axis], tuple(options)))
    return [Question(fact, axis, texts, truths) for (fact, axis), (texts, truths) in found.items()]


def accuracy(answers: list[str], truths: list[str]) -> float:
    return sum(answer == truth for answer, truth in zip(answers, truths, strict=True)) / len(truths)


def laya_reader(checkpoint: str, device: str) -> Reader:
    networks: dict[str, LayaNetwork] = {}

    def read(texts: list[str], question: str, options: dict[str, str]) -> list[str]:
        key = ",".join(options)
        if key not in networks:
            networks[key] = LayaNetwork(checkpoint, options, device)
        network = networks[key]
        option_vectors, _ = network.features(texts, [question] * len(texts))
        with torch.no_grad():
            choices = network.scorer(option_vectors).squeeze(-1).argmax(-1).tolist()
        return [list(options)[choice] for choice in choices]

    return read


def d1_reader(model: object) -> Reader:
    def read(texts: list[str], question: str, options: dict[str, str]) -> list[str]:
        asked = {"type": "choice", "instructions": question, "criteria": options}
        rows = model.probabilities_batch([(text, [asked]) for text in texts])
        return [list(options)[max(range(len(options)), key=row[0].__getitem__)] for row in rows]

    return read


def compare(asked: list[Question], laya: Reader, d1: Reader) -> list[Score]:
    scores = []
    for question in asked:
        options = AXES[question.axis]
        majority = Counter(question.truths).most_common(1)[0][1] / len(question.truths)
        laya_score = accuracy(laya(question.texts, question.text, options), question.truths)
        d1_score = accuracy(d1(question.texts, question.text, options), question.truths)
        scores.append(Score(question.fact, question.axis, len(question.truths), majority, laya_score, d1_score))
    return scores


def readers_compared(wrapper: str, config: Path, decisions: int, seed: int) -> None:
    settings = load_config(config)
    setup_logging(settings.log_level)
    device = configure_accelerator()
    samples = collect_samples(wrapper_class(wrapper), settings, decisions, seed)
    d1 = AutoModel.from_pretrained(D1_CHECKPOINT, trust_remote_code=True).to(device).eval()
    for score in compare(questions(samples), laya_reader(settings.laya.checkpoint, device), d1_reader(d1)):
        logger.info(
            f"{score.fact:>16}.{score.axis:<5} n={score.samples:<5} majority {score.majority:.2f} "
            f"laya {score.laya:.2f} d1-omni {score.d1:.2f}"
        )


if __name__ == "__main__":
    typer.run(readers_compared)
