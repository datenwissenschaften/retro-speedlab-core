import json
from pathlib import Path

from datenwissenschaften.route.position import Position

CELL_SIZE = 16

Cell = tuple[int, int, int]
Remaining = dict[Cell, int]


def cell_of(position: Position) -> Cell:
    return position.area, position.x // CELL_SIZE, position.y // CELL_SIZE


def load_route(path: Path) -> Remaining:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(area, x, y): left for area, x, y, _, left in rows}


def changed_at(path: Path) -> int | None:
    return path.stat().st_mtime_ns if path.is_file() else None


class RouteMap:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.route: Remaining = {}
        self.loaded: int | None = None

    def remaining(self, position: Position | None) -> int | None:
        if position is None:
            return None
        changed = changed_at(self.path)
        if changed != self.loaded:
            self.route, self.loaded = ({} if changed is None else load_route(self.path)), changed
        cell = cell_of(position)
        return self.route[cell] if cell in self.route else None
