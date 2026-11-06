from collections import Counter
from datetime import UTC, datetime

import wandb

from datenwissenschaften.speedrun.journal import JournalStore, Split
from datenwissenschaften.speedrun.movie import Segment

SECONDS_PER_HOUR = 3600


class SearchStats:
    def __init__(self, experiment: wandb.Run, journal: JournalStore, opening: str) -> None:
        self.experiment = experiment
        self.journal = journal
        self.opening = opening
        self.events: Counter[str] = Counter()
        self.frames_saved = 0

    def run(self, segments: list[Segment], found: list[Split], seconds: float) -> None:
        beaten = sum(split.level != self.opening for split in found)
        reached = {"run/reached": segments[-1].state} if segments else {}
        levels = {f"levels/{split.level}": split.level_seconds for split in found}
        self.log(
            {"run/seconds": seconds, "run/pieces": len(segments), "run/levels_beaten": beaten, **reached, **levels}
        )

    def event(self, kind: str) -> None:
        self.events[kind] += 1
        self.log({f"search/{kind}": self.events[kind]})

    def trimmed(self, frames: int) -> None:
        self.frames_saved += frames
        self.log({"search/frames_saved": self.frames_saved})

    def log(self, metrics: dict[str, float | int | str]) -> None:
        progressed = self.journal.progressed()
        drought = (
            {}
            if progressed is None
            else {"search/hours_since_progress": (datetime.now(UTC) - progressed).total_seconds() / SECONDS_PER_HOUR}
        )
        self.experiment.log({**metrics, **drought})
