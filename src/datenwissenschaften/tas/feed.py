import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import msgspec
import wandb

from datenwissenschaften.tas.journal import JournalStore
from datenwissenschaften.tas.live import clear_reports, read_reports
from datenwissenschaften.tas.live_search import SearchRound, search_field
from datenwissenschaften.tas.movie import Segment, frames
from datenwissenschaften.tas.report import SECTION, Course, report, splits
from datenwissenschaften.tas.routes import PartProgress
from datenwissenschaften.tas.stuck import load_stuck
from datenwissenschaften.ui.telemetry import publish_metadata

PUBLISH_SECONDS = 5.0
REEL_ORDER = "number"


class SpeedrunFeed:
    def __init__(
        self, course: Course, journal: JournalStore, live: Path, stuck: Path, lab_marker: Path, experiment: wandb.Run
    ) -> None:
        self.course = course
        self.journal = journal
        self.live = live
        self.stuck = stuck
        self.lab_marker = lab_marker
        self.experiment = experiment
        self.events: Counter[str] = Counter()
        self.lock = threading.Lock()
        self.summary: dict[str, Any] = {}
        self.searching: str | None = None
        self.search: SearchRound | None = None
        self.screened: dict[str, dict[str, Any] | None] = {"movie": None, "compare": None}
        self.practice: dict[str, dict[str, Any]] = {}
        self.reel: dict[str, dict[str, Any]] = {}
        self.lab_running = lab_marker.exists()
        live.mkdir(parents=True, exist_ok=True)

    def show(self, segments: list[Segment], searching: str) -> None:
        found = splits(self.course, segments)
        seconds = frames(segments) / self.course.frame_rate
        firsts = self.journal.track(found, seconds)
        levels = {f"levels/{split.level}": split.level_seconds for split in found}
        self.experiment.log({"run/seconds": seconds, "run/pieces": len(segments), **levels})
        summary = report(self.course, segments, found, firsts, searching, load_stuck(self.stuck))
        with self.lock:
            self.summary, self.searching = summary, searching
        self.publish()

    def shown(self, name: str, field: dict[str, Any]) -> None:
        with self.lock:
            self.screened = {**self.screened, name: field}
        self.publish()

    def reeled(self, level: str, field: dict[str, Any]) -> None:
        with self.lock:
            self.reel = {**self.reel, level: field}
        self.publish()

    def unreeled(self, level: str) -> None:
        with self.lock:
            self.reel = {name: video for name, video in self.reel.items() if name != level}

    def practised(self, progress: dict[str, PartProgress]) -> None:
        practice = {level: self.level_progress(part) for level, part in progress.items()}
        with self.lock:
            self.practice = practice
        self.publish()

    def level_progress(self, part: PartProgress) -> dict[str, Any]:
        return {**msgspec.to_builtins(part), "seconds": part.frames / self.course.frame_rate}

    def event(self, kind: str, state: str | None, frames: int | None) -> None:
        self.journal.event(kind, state, frames)
        self.events[kind] += 1
        shown = {} if frames is None else {f"search/{kind}_frames": frames}
        self.experiment.log({f"search/{kind}": self.events[kind], **shown})

    def begin(self, search: SearchRound) -> None:
        clear_reports(self.live)
        with self.lock:
            self.search = search
        self.publish()

    def end(self) -> None:
        with self.lock:
            self.search = None
        clear_reports(self.live)
        self.publish()

    def publish(self) -> None:
        with self.lock:
            summary, search, screened, practice = self.summary, self.search, self.screened, self.practice
            reel = sorted(self.reel.values(), key=lambda video: video[REEL_ORDER])
        if not summary:
            return
        reports = [] if search is None else read_reports(self.live)
        now = datetime.now(UTC)
        live = {"search": search_field(search, reports, now), "updated_at": now.isoformat(timespec="seconds")}
        fields = {**summary, **screened, "practice": practice, "reel": reel}
        publish_metadata(SECTION, {**fields, **self.journal.fields(), **live})

    def notice_lab(self) -> None:
        running = self.lab_marker.exists()
        if running and not self.lab_running:
            stuck = load_stuck(self.stuck)
            self.event("lab", self.searching if stuck is None else stuck.state, None)
        self.lab_running = running

    def publish_every(self, stopped: threading.Event) -> None:
        while not stopped.wait(PUBLISH_SECONDS):
            self.notice_lab()
            self.publish()
