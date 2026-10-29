from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from datenwissenschaften.tas.live import WorkerReport


@dataclass(slots=True, frozen=True)
class SearchRound:
    state: str
    round: int
    rounds: int
    started_at: datetime
    seconds: float


def search_round(state: str, index: int, rounds: int, seconds: float) -> SearchRound:
    return SearchRound(state, index, rounds, datetime.now(UTC), seconds)


def search_field(search: SearchRound | None, reports: list[WorkerReport], now: datetime) -> dict[str, Any] | None:
    if search is None:
        return None
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
