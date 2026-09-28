import random
import zlib
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

    def encode(
        self, states: list[str], questions: list[str], device: torch.device
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        items = [[self._item(state, question)] for state, question in zip(states, questions, strict=True)]
        batch = collate_items(items, self.tokenizer.pad_token_id)
        orders = torch.tensor([group[0]["order"] for group in items], device=device)
        return {key: batch[key].to(device) for key in BATCH_KEYS}, orders

    def _item(self, state: str, question: str) -> dict[str, Any]:
        definition = {"t": "choice", "ins": question, "crit": self.options}
        order = self._order(state, question)
        ids, markers = build_sequence(
            self.tokenizer, state, definition, self.max_len, self.head_max_len, option_order=order
        )
        return {"ids": ids, "markers": markers, "qtype": QTYPES["choice"], "order": order}

    def _order(self, state: str, question: str) -> list[int]:
        order = list(range(len(self.options)))
        random.Random(zlib.crc32(f"{question}\n{state}".encode())).shuffle(order)
        return order
