from datenwissenschaften.route.position import Position
from datenwissenschaften.route.state_route import Cell, cell_of

MAX_STILL_SECONDS = 30.0


class Stillness:
    def __init__(self, max_still_frames: int) -> None:
        self.max_still_frames = max_still_frames
        self.visited: set[Cell] = set()
        self.still_frames = 0

    def restart(self) -> None:
        self.visited = set()
        self.still_frames = 0

    def stalled(self, position: Position | None) -> bool:
        if position is None:
            return False
        cell = cell_of(position)
        if cell in self.visited:
            self.still_frames += 1
        else:
            self.visited.add(cell)
            self.still_frames = 0
        return self.still_frames >= self.max_still_frames
