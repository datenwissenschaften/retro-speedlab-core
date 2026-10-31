import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from datenwissenschaften.tas.journal import JournalStore
from datenwissenschaften.tas.live import clear_reports, read_reports
from datenwissenschaften.tas.live_search import SearchRound, search_field
from datenwissenschaften.tas.movie import Segment, frames
from datenwissenschaften.tas.report import SECTION, Course, report, splits
from datenwissenschaften.tas.stuck import load_stuck
from datenwissenschaften.ui.telemetry import publish_metadata

PUBLISH_SECONDS = 5.0


class SpeedrunFeed:
    def __init__(self, course: Course, journal: JournalStore, live: Path, stuck: Path, lab_marker: Path) -> None:
        self.course = course
        self.journal = journal
        self.live = live
        self.stuck = stuck
        self.lab_marker = lab_marker
        self.lock = threading.Lock()
        self.summary: dict[str, Any] = {}
        self.searching: str | None = None
        self.search: SearchRound | None = None
        self.screened: dict[str, dict[str, Any] | None] = {"movie": None, "compare": None}
        self.lab_running = lab_marker.exists()
        live.mkdir(parents=True, exist_ok=True)

    def show(self, segments: list[Segment], searching: str) -> None:
        found = splits(self.course, segments)
        firsts = self.journal.track(found, frames(segments) / self.course.frame_rate)
        summary = report(self.course, segments, found, firsts, searching, load_stuck(self.stuck))
        with self.lock:
            self.summary, self.searching = summary, searching
        self.publish()

    def shown(self, name: str, field: dict[str, Any]) -> None:
        with self.lock:
            self.screened = {**self.screened, name: field}
        self.publish()

    def event(self, kind: str, state: str | None, frames: int | None) -> None:
        self.journal.event(kind, state, frames)

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
            summary, search, screened = self.summary, self.search, self.screened
        if not summary:
            return
        reports = [] if search is None else read_reports(self.live)
        now = datetime.now(UTC)
        live = {"search": search_field(search, reports, now), "updated_at": now.isoformat(timespec="seconds")}
        publish_metadata(SECTION, {**summary, **screened, **self.journal.fields(), **live})

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
