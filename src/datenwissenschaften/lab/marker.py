import time
from dataclasses import dataclass
from pathlib import Path

import msgspec


@dataclass(slots=True, frozen=True)
class Score:
    agent: str
    wins: int
    progress: float
    decisions: float


@dataclass(slots=True, frozen=True)
class DuelResult:
    winner: str
    confidence: float
    scores: list[Score]


@dataclass(slots=True, frozen=True)
class AgentBranch:
    log: str
    worktree: str


@dataclass(slots=True, frozen=True)
class LabMarker:
    until: float
    phase: str
    base: str
    agents: dict[str, AgentBranch]
    duel: DuelResult | None


def read_marker(marker: Path) -> LabMarker | None:
    if not marker.is_file():
        return None
    lab = msgspec.json.decode(marker.read_bytes(), type=LabMarker)
    return lab if lab.until > time.time() else None
