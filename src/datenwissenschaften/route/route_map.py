import json
from collections.abc import Callable
from pathlib import Path

from datenwissenschaften.route.position import Position

CELL_SIZE = 8

Cell = tuple[int, int, int, int, bool]
Remaining = dict[Cell, int]


def cell_of(position: Position) -> Cell:
    size = CELL_SIZE
    return position.area, position.x // size, position.y // size, position.height // size, position.airborne


def load_route(path: Path) -> Remaining:
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(area, x, y, height, bool(air)): left for area, x, y, height, air, _, left in rows}


class RouteMap:
    def __init__(self, path: Callable[[str], Path]) -> None:
        self.path = path
        self.routes: dict[str, Remaining] = {}

    def remaining(self, state: str, position: Position | None) -> int | None:
        if position is None:
            return None
        if state not in self.routes:
            self.routes[state] = load_route(self.path(state))
        cell = cell_of(position)
        return self.routes[state][cell] if cell in self.routes[state] else None
