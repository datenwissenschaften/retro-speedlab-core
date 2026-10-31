import json
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
