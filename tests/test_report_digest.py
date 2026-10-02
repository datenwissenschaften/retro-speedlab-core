import json
import os
from pathlib import Path

import pytest

from datenwissenschaften.ui.report_digest import ReportDigest

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


def test_the_summary_is_written_next_to_the_newest_report_and_reused(tmp_path: Path):
    (tmp_path / "2026-10-01.md").write_text("old", encoding="utf-8")
    (tmp_path / "2026-10-02.md").write_text("new", encoding="utf-8")
    summarizer = CountingSummarizer()
    digest = ReportDigest(tmp_path, summarizer)

    first = digest.latest()
    again = ReportDigest(tmp_path, summarizer).latest()

    assert first == again == {"name": "2026-10-02.md", "headline": SUMMARY[0], "lines": SUMMARY[1:]}
    assert json.loads((tmp_path / "2026-10-02.summary.json").read_text(encoding="utf-8")) == first
    assert summarizer.reports == ["new"]


def test_a_report_changed_after_its_summary_is_summarized_again(tmp_path: Path):
    report = tmp_path / "2026-10-02.md"
    report.write_text("morning", encoding="utf-8")
    summarizer = CountingSummarizer()
    digest = ReportDigest(tmp_path, summarizer)
    digest.latest()
    summary = tmp_path / "2026-10-02.summary.json"
    os.utime(summary, (report.stat().st_mtime - 60, report.stat().st_mtime - 60))
    report.write_text("evening", encoding="utf-8")

    digest.latest()

    assert summarizer.reports == ["morning", "evening"]


def test_without_a_report_there_is_no_summary(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        ReportDigest(tmp_path, CountingSummarizer()).latest()
