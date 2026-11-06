import json
from pathlib import Path

from datenwissenschaften.route.position import Position

CELL_SIZE = 16
LEFT_COLUMN = 4
NEEDS_FROM = 5

Cell = tuple[int, int, int]
Needs = tuple[int, ...]
Remaining = dict[Cell, list[tuple[Needs, int]]]


def cell_of(position: Position) -> Cell:
    return position.area, position.x // CELL_SIZE, position.y // CELL_SIZE


def load_route(path: Path) -> list[tuple[Cell, Needs, int]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [((row[0], row[1], row[2]), tuple(row[NEEDS_FROM:]), row[LEFT_COLUMN]) for row in rows]


def merged(paths: tuple[Path, ...]) -> Remaining:
    route: Remaining = {}
    for path in paths:
        if not path.is_file():
            continue
        for cell, needs, left in load_route(path):
            route.setdefault(cell, []).append((needs, left))
    return route


def changed_at(path: Path) -> int | None:
    return path.stat().st_mtime_ns if path.is_file() else None


def met(have: Needs, needs: Needs) -> bool:
    return len(have) >= len(needs) and all(held >= needed for held, needed in zip(have, needs, strict=False))


def left_in(route: Remaining, cell: Cell | None, have: Needs) -> int | None:
    if cell is None or cell not in route:
        return None
    return min((left for needs, left in route[cell] if met(have, needs)), default=None)


class RouteMap:
    def __init__(self, paths: tuple[Path, ...]) -> None:
        self.paths = paths
        self.route: Remaining = {}
        self.loaded: tuple[int | None, ...] = tuple(None for _ in paths)

    def reload(self) -> Remaining:
        changed = tuple(changed_at(path) for path in self.paths)
        if changed != self.loaded:
            self.route, self.loaded = merged(self.paths), changed
        return self.route

    def remaining(self, position: Position | None, have: Needs) -> int | None:
        if position is None:
            return None
        return left_in(self.reload(), cell_of(position), have)
