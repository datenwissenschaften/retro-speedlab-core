import json
import os
from pathlib import Path
from typing import Any

import pytest

from datenwissenschaften.ui import summaries
from datenwissenschaften.ui.summaries import CachedSummary, ReportDigest

LINE_BREAK = chr(10)
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


def digest_of(tmp_path: Path, summarizer: CountingSummarizer) -> ReportDigest:
    return ReportDigest(tmp_path, CachedSummary(summarizer))


def summaries_of(digest: ReportDigest, published: set[str]) -> list[dict[str, Any]]:
    return [digest.of(report) for report in digest.due(published)]


def test_the_summary_is_written_next_to_the_newest_report_and_reused(tmp_path: Path):
    (tmp_path / "2026-10-01.md").write_text("old", encoding="utf-8")
    (tmp_path / "2026-10-02.md").write_text("new", encoding="utf-8")
    summarizer = CountingSummarizer()
    published = {"2026-10-01.md"}

    first = summaries_of(digest_of(tmp_path, summarizer), published)
    again = summaries_of(digest_of(tmp_path, summarizer), published)

    assert first == again == [{"name": "2026-10-02.md", "headline": SUMMARY[0], "lines": SUMMARY[1:]}]
    assert json.loads((tmp_path / "2026-10-02.summary.json").read_text(encoding="utf-8")) == first[0]
    assert summarizer.reports == ["new"]


def test_the_newest_report_and_every_report_the_website_lacks_are_due_newest_first(tmp_path: Path):
    for name in ("2026-10-01.md", "2026-10-02.md", "2026-10-03.md", "2026-10-04.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    due = digest_of(tmp_path, CountingSummarizer()).due({"2026-10-02.md", "2026-10-04.md"})

    assert [report.name for report in due] == ["2026-10-04.md", "2026-10-03.md", "2026-10-01.md"]


def test_a_report_changed_after_its_summary_is_summarized_again(tmp_path: Path):
    report = tmp_path / "2026-10-02.md"
    report.write_text("morning", encoding="utf-8")
    summarizer = CountingSummarizer()
    digest = digest_of(tmp_path, summarizer)
    digest.of(report)
    summary = tmp_path / "2026-10-02.summary.json"
    os.utime(summary, (report.stat().st_mtime - 60, report.stat().st_mtime - 60))
    report.write_text("evening", encoding="utf-8")

    digest.of(report)

    assert summarizer.reports == ["morning", "evening"]


def test_without_a_report_nothing_is_due(tmp_path: Path):
    assert digest_of(tmp_path, CountingSummarizer()).due(set()) == []


def test_a_timed_report_of_the_day_is_newer_than_the_dated_one(tmp_path: Path):
    names = ("2026-10-02T2130.md", "2026-10-03.md", "2026-10-03T0930.md")
    for name in names:
        (tmp_path / name).write_text(name, encoding="utf-8")

    assert [report.name for report in digest_of(tmp_path, CountingSummarizer()).due(set(names))] == [
        "2026-10-03T0930.md"
    ]


class FakeModels:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"data": [{"id": model} for model in self.ids]}


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def replying(monkeypatch, *contents: str) -> list[str]:
    replies, asked = iter(contents), []

    def post(url, **kwargs):
        asked.append(url)
        return FakeResponse(next(replies))

    monkeypatch.setattr(summaries.httpx, "post", post)
    monkeypatch.setattr(summaries.httpx, "get", lambda url, **kwargs: FakeModels(["model"]))
    monkeypatch.setattr(summaries.Summarizer.summarize.retry, "sleep", lambda seconds: None)
    return asked


def test_a_headline_with_two_lines_is_enough(monkeypatch):
    asked = replying(monkeypatch, "Laya starts\nMenu mastered\nWatch the pibbleys")

    lines = summaries.Summarizer("rules", ("model",), "key").summarize("report")

    assert lines == ["Laya starts", "Menu mastered", "Watch the pibbleys"]
    assert len(asked) == 1


def test_a_too_short_summary_is_asked_again(monkeypatch):
    asked = replying(monkeypatch, "Only a headline", "\n".join(SUMMARY))

    assert summaries.Summarizer("rules", ("model",), "key").summarize("report") == SUMMARY
    assert len(asked) == 2


def test_the_summary_gives_up_after_its_attempts(monkeypatch):
    asked = replying(monkeypatch, *["Only a headline"] * summaries.SUMMARY_ATTEMPTS)

    with pytest.raises(ValueError, match="summary lines"):
        summaries.Summarizer("rules", ("model",), "key").summarize("report")
    assert len(asked) == summaries.SUMMARY_ATTEMPTS


def test_models_openrouter_withdrew_are_left_out_of_the_request(monkeypatch):
    sent = []
    monkeypatch.setattr(summaries.httpx, "get", lambda url, **kwargs: FakeModels(["kept", "other"]))
    monkeypatch.setattr(
        summaries.httpx,
        "post",
        lambda url, **kwargs: sent.append(kwargs["json"]["models"]) or FakeResponse(LINE_BREAK.join(SUMMARY)),
    )

    summaries.Summarizer("rules", ("gone", "kept"), "key").summarize("report")

    assert sent == [["kept"]]


def test_without_any_offered_model_there_is_no_summary(monkeypatch):
    monkeypatch.setattr(summaries.httpx, "get", lambda url, **kwargs: FakeModels(["other"]))
    monkeypatch.setattr(summaries.Summarizer.summarize.retry, "sleep", lambda seconds: None)

    with pytest.raises(ValueError):
        summaries.Summarizer("rules", ("gone",), "key").summarize("report")


def test_a_rate_limited_summary_is_not_asked_again(monkeypatch):
    asked = []
    request = summaries.httpx.Request("POST", summaries.OPENROUTER_URL)

    def post(url: str, **kwargs) -> summaries.httpx.Response:
        asked.append(url)
        return summaries.httpx.Response(summaries.httpx.codes.TOO_MANY_REQUESTS, request=request)

    monkeypatch.setattr(summaries.httpx, "post", post)
    monkeypatch.setattr(summaries.httpx, "get", lambda url, **kwargs: FakeModels(["model"]))
    monkeypatch.setattr(summaries.Summarizer.summarize.retry, "sleep", lambda seconds: None)

    with pytest.raises(summaries.httpx.HTTPStatusError):
        summaries.Summarizer("rules", ("model",), "key").summarize("report")
    assert len(asked) == 1
