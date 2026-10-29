from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from datenwissenschaften.route.position import Position
from datenwissenschaften.tas.live import MAP_CELLS, WorkerReport
from datenwissenschaften.tas.runner import MAP_PIXELS, Spot, spot


@dataclass(slots=True, frozen=True)
class SearchRound:
    state: str
    round: int
    rounds: int
    started_at: datetime
    seconds: float
    start: Position | None


def search_round(state: str, index: int, rounds: int, seconds: float, start: Position | None) -> SearchRound:
    return SearchRound(state, index, rounds, datetime.now(UTC), seconds, start)


def search_field(search: SearchRound, reports: list[WorkerReport], now: datetime) -> dict[str, Any]:
    played = sum(report.frames for report in reports)
    elapsed = (now - search.started_at).total_seconds()
    best = [report.best_frames for report in reports if report.best_frames is not None]
    return {
        "state": search.state,
        "round": search.round + 1,
        "rounds": search.rounds,
        "started_at": search.started_at.isoformat(timespec="seconds"),
        "ends_at": (search.started_at + timedelta(seconds=search.seconds)).isoformat(timespec="seconds"),
        "workers": [
            {"seed": report.seed, "cells": report.cells, "frames": report.frames, "best_frames": report.best_frames}
            for report in reports
        ],
        "cells": sum(report.cells for report in reports),
        "frames": played,
        "frames_per_second": played / elapsed if elapsed > 0 else 0.0,
        "best_frames": min(best, default=None),
    }


def map_field(search: SearchRound, reports: list[WorkerReport], route: list[Spot]) -> dict[str, Any] | None:
    visits: Counter[Spot] = Counter()
    for report in reports:
        for area, x, y, count in report.visits:
            visits[area, x, y] += count
    if search.start is None and not visits:
        return None
    area = search.start.area if search.start is not None else visits.most_common(1)[0][0][0]
    cells = [[x, y, count] for (cell_area, x, y), count in visits.most_common() if cell_area == area]
    return {
        "state": search.state,
        "area": area,
        "cell_pixels": MAP_PIXELS,
        "cells": cells[:MAP_CELLS],
        "route": [[x, y] for route_area, x, y in route if route_area == area],
        "player": None if search.start is None else list(spot(search.start)[1:]),
    }


def position_field(position: Position | None) -> dict[str, int] | None:
    if position is None:
        return None
    return {"area": position.area, "x": position.x, "y": position.y, "height": position.height}


def live_fields(
    search: SearchRound | None, reports: list[WorkerReport], route: list[Spot], now: datetime
) -> dict[str, Any]:
    if search is None:
        return {"search": None, "map": None, "ram": None, "position": None}
    shown = next((report for report in reports if report.shows), None)
    return {
        "search": search_field(search, reports, now),
        "map": map_field(search, reports, route),
        "ram": None if shown is None else shown.ram,
        "position": None if shown is None else position_field(shown.position),
    }
