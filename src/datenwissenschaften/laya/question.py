from typing import Any

import torch
from laya.common import QTYPES, build_sequence, collate_items

BATCH_KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


class LayaQuestion:
    def __init__(self, tokenizer: Any, config: dict[str, Any], options: dict[str, str]) -> None:
        self.tokenizer = tokenizer
        self.max_len = int(config["max_len"])
        self.head_max_len = int(config["head_max_len"])
        self.options = dict(options)

    def require_fits(self, question: str) -> None:
        if len(self._item("", question)["markers"]) != len(self.options):
            raise ValueError(f"Question {question!r} exceeds Laya's budget of {self.head_max_len} tokens.")

    def encode(self, states: list[str], questions: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        items = [[self._item(state, question)] for state, question in zip(states, questions, strict=True)]
        batch = collate_items(items, self.tokenizer.pad_token_id)
        return {key: batch[key].to(device) for key in BATCH_KEYS}

    def _item(self, state: str, question: str) -> dict[str, Any]:
        definition = {"t": "choice", "ins": question, "crit": self.options}
        ids, markers = build_sequence(self.tokenizer, state, definition, self.max_len, self.head_max_len)
        return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"]}
