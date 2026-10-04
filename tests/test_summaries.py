import json
import os
from pathlib import Path

import pytest

from datenwissenschaften.ui.summaries import CachedSummary, LabSummaries

SUMMARY = [
    "Laya Meets The Nibbleys",
    "Today's update shows Laya the nearest nibbley.",
    "It eats two per run.",
    "Watch it chase them.",
]


class CountingSummarizer:
    def __init__(self) -> None:
        self.reports: list[str] = []

    def summarize(self, report: str) -> list[str]:
        self.reports.append(report)
        return SUMMARY


def summaries(tmp_path: Path, summarizer: CountingSummarizer) -> LabSummaries:
    cached = CachedSummary(summarizer)
    return LabSummaries(tmp_path, tmp_path / "HINT.md", cached, cached)


def test_the_summary_is_written_next_to_the_newest_report_and_reused(tmp_path: Path):
    (tmp_path / "2026-10-01.md").write_text("old", encoding="utf-8")
    (tmp_path / "2026-10-02.md").write_text("new", encoding="utf-8")
    summarizer = CountingSummarizer()
    digest = summaries(tmp_path, summarizer)

    first = digest.latest_report()
    again = summaries(tmp_path, summarizer).latest_report()

    assert first == again == {"name": "2026-10-02.md", "headline": SUMMARY[0], "lines": SUMMARY[1:]}
    assert json.loads((tmp_path / "2026-10-02.summary.json").read_text(encoding="utf-8")) == first
    assert summarizer.reports == ["new"]


def test_a_report_changed_after_its_summary_is_summarized_again(tmp_path: Path):
    report = tmp_path / "2026-10-02.md"
    report.write_text("morning", encoding="utf-8")
    summarizer = CountingSummarizer()
    digest = summaries(tmp_path, summarizer)
    digest.latest_report()
    summary = tmp_path / "2026-10-02.summary.json"
    os.utime(summary, (report.stat().st_mtime - 60, report.stat().st_mtime - 60))
    report.write_text("evening", encoding="utf-8")

    digest.latest_report()

    assert summarizer.reports == ["morning", "evening"]


def test_without_a_report_there_is_no_summary(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        summaries(tmp_path, CountingSummarizer()).latest_report()


def test_a_timed_report_of_the_day_is_newer_than_the_dated_one(tmp_path: Path):
    for name in ("2026-10-02T2130.md", "2026-10-03.md", "2026-10-03T0930.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    summarizer = CountingSummarizer()

    assert summaries(tmp_path, summarizer).latest_report()["name"] == "2026-10-03T0930.md"
    assert summarizer.reports == ["2026-10-03T0930.md"]


def test_the_hints_are_summarized_next_to_the_hint_file_and_resummarized_when_edited(tmp_path: Path):
    hints = tmp_path / "HINT.md"
    hints.write_text("The door opens at weight four.", encoding="utf-8")
    summarizer = CountingSummarizer()
    lab = summaries(tmp_path, summarizer)

    first = lab.hint()
    os.utime(tmp_path / "HINT.summary.json", (hints.stat().st_mtime - 60, hints.stat().st_mtime - 60))
    hints.write_text("The discs are enemies.", encoding="utf-8")
    lab.hint()

    assert first == {"name": "HINT.md", "headline": SUMMARY[0], "lines": SUMMARY[1:]}
    assert summarizer.reports == ["The door opens at weight four.", "The discs are enemies."]


def test_without_a_hint_file_there_is_no_hint_summary(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        summaries(tmp_path, CountingSummarizer()).hint()
