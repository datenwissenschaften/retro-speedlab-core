from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import Cell, cell_of

LEVEL_GAP_FRAMES = 60

Step = tuple[int, int, int, int]
Timed = tuple[Step, int]


def stretches(steps: list[Step]) -> list[list[Step]]:
    found: list[list[Step]] = []
    for step in steps:
        if found and step[0] - found[-1][-1][0] <= LEVEL_GAP_FRAMES:
            found[-1].append(step)
        else:
            found.append([step])
    return found


def timed_rows(timed: list[Timed]) -> list[list[int]]:
    remaining: dict[Cell, tuple[int, int]] = {}
    for (frame, area, x, y), end in timed:
        cell = cell_of(Position(area, x, y, 0, False))
        if cell not in remaining or end - frame < remaining[cell][1]:
            remaining[cell] = (frame, end - frame)
    return [[*cell, frame, left] for cell, (frame, left) in sorted(remaining.items())]


def route_rows(steps: list[Step]) -> list[list[int]]:
    return timed_rows([(step, stretch[-1][0]) for stretch in stretches(steps) for step in stretch])
