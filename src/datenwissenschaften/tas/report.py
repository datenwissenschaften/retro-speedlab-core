from datetime import UTC, datetime

from datenwissenschaften.settings import SpeedrunRecord
from datenwissenschaften.tas.movie import Segment

SECTION = "speedrun"


def splits(segments: list[Segment], levels: dict[str, tuple[str, ...]], first: str, frame_rate: float) -> list[dict]:
    leaving = {states[-1]: level for level, states in levels.items()}
    found, frames, origin = [], 0, first
    for segment in segments:
        frames += len(segment.pressed)
        if origin in leaving:
            found.append({"level": leaving[origin], "seconds": frames / frame_rate})
        origin = segment.state
    return found


def report(
    segments: list[Segment],
    levels: dict[str, tuple[str, ...]],
    first: str,
    frame_rate: float,
    records: tuple[SpeedrunRecord, ...],
    searching: str,
) -> dict:
    frames = sum(len(segment.pressed) for segment in segments)
    return {
        "frames": frames,
        "seconds": frames / frame_rate,
        "frame_rate": frame_rate,
        "segments": [{"state": segment.state, "frames": len(segment.pressed)} for segment in segments],
        "splits": splits(segments, levels, first, frame_rate),
        "records": [{"name": record.name, "seconds": record.seconds} for record in records],
        "reached": segments[-1].state if segments else first,
        "won": bool(segments) and segments[-1].won,
        "searching": searching,
        "updated_at": datetime.now(UTC).isoformat(),
    }
