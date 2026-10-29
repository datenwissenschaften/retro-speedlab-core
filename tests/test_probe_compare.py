from types import SimpleNamespace

import torch
from fakes import write_config

from datenwissenschaften.probe import compare
from datenwissenschaften.probe.samples import Sample
from datenwissenschaften.states.facts import Offset

SAMPLES = [
    Sample("Play", {"to_target": Offset(4, -2), "lives": 3}, "target right and up", "Which move?"),
    Sample("Play", {"to_target": Offset(-3, 0), "lives": 3}, "target left", "Which move?"),
    Sample("Play", {"to_target": Offset(0, 5), "lives": 3}, "target below", "Which move?"),
]


def truthful(texts: list[str], question: str, options: dict[str, str]) -> list[str]:
    words = {"right": "right", "left": "left", "below": "down", "up": "up"}
    return [next((words[word] for word in words if word in text and words[word] in options), "here") for text in texts]


def test_every_offset_becomes_a_left_right_and_an_up_down_question():
    asked = compare.questions(SAMPLES)

    assert [(question.fact, question.axis) for question in asked] == [("to_target", "right"), ("to_target", "down")]
    assert asked[0].truths == ["right", "left", "here"]
    assert asked[1].truths == ["up", "here", "down"]


def test_readers_are_scored_against_the_truth_and_the_majority_answer():
    scores = compare.compare(compare.questions(SAMPLES), truthful, lambda texts, question, options: ["here"] * 3)

    assert [(score.laya, score.d1) for score in scores] == [(1.0, 1 / 3), (1.0, 1 / 3)]
    assert scores[0].majority == 1 / 3


class FakeNetwork:
    def __init__(self, checkpoint: str, options: dict[str, str], device: str) -> None:
        self.options = list(options)

    def features(self, texts: list[str], questions: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        chosen = torch.zeros(len(texts), len(self.options), 1)
        chosen[:, -1] = 1.0
        return chosen, chosen[:, 0]

    def scorer(self, options: torch.Tensor) -> torch.Tensor:
        return options


class FakeD1:
    def probabilities_batch(self, requests: list[tuple]) -> list[list[list[float]]]:
        return [[[0.1, 0.8, 0.1]] for _ in requests]

    def to(self, device: str) -> "FakeD1":
        return self

    def eval(self) -> "FakeD1":
        return self


def test_both_readers_answer_from_their_own_model_and_the_command_logs_every_fact(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "LayaNetwork", FakeNetwork)
    monkeypatch.setattr(compare, "collect_samples", lambda wrapper_cls, settings, decisions, seed: SAMPLES)
    monkeypatch.setattr(
        compare, "AutoModel", SimpleNamespace(from_pretrained=lambda name, trust_remote_code, dtype: FakeD1())
    )
    monkeypatch.setattr(compare, "configure_accelerator", lambda: "cpu")
    logged: list[str] = []
    monkeypatch.setattr(compare.logger, "info", logged.append)
    options = compare.AXES["right"]

    assert compare.laya_reader("laya", "cpu")(["a", "b"], "Where?", options) == ["right", "right"]
    assert compare.d1_reader(FakeD1())(["a"], "Where?", options) == ["here"]
    compare.readers_compared("fakes:FakeWrapper", write_config(tmp_path), 10, 0)
    assert len(logged) == 2 and "d1-omni 0.33" in logged[0]
