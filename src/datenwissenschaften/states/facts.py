import json
from dataclasses import dataclass
from typing import Any

Facts = dict[str, Any]


@dataclass(slots=True, frozen=True)
class Offset:
    right: int
    down: int

    def __str__(self) -> str:
        return f"{_side(self.right, 'left', 'right')}, {_side(self.down, 'above', 'below')}"


def render_facts(facts: Facts) -> str:
    return json.dumps(facts, default=str)


def _side(distance: int, negative: str, positive: str) -> str:
    if distance == 0:
        return "level"
    return f"{abs(distance)} {negative if distance < 0 else positive}"
