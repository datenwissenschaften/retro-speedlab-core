import json
from pathlib import Path

from datenwissenschaften.route.position import Position

CELL_SIZE = 16

Cell = tuple[int, int, int]
Remaining = dict[Cell, int]


def cell_of(position: Position) -> Cell:
    return position.area, position.x // CELL_SIZE, position.y // CELL_SIZE


def load_route(path: Path) -> Remaining:
    if not path.is_file():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(area, x, y): left for area, x, y, _, left in rows}


class RouteMap:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.route: Remaining | None = None

    def remaining(self, position: Position | None) -> int | None:
        if position is None:
            return None
        if self.route is None:
            self.route = load_route(self.path)
        cell = cell_of(position)
        return self.route[cell] if cell in self.route else None
