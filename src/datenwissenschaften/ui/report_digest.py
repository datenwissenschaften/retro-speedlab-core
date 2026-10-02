import json
import os
import threading
from pathlib import Path
from typing import Any

import httpx

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.reports import list_reports, read_report

OPENROUTER_KEY = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60
MAX_TOKENS = 300
SUMMARY_LINES = 4
SUMMARY_SUFFIX = ".summary.json"
DECORATION = " \t-*•#>\"'`"
INSTRUCTIONS = (
    "You write the lab update card on a live stream where Laya, an AI, teaches itself to play {game} by trial "
    "and error. Read today's lab report and write exactly four lines for the viewers. Line 1: a catchy headline "
    "of at most six words. Lines 2 to 4: one short, lively sentence each, at most 14 words: what changed today, "
    "how Laya is doing, and what to watch for next. Plain text only, without markdown, bullets, numbering, "
    "emojis or quotes. Never mention Claude, coding agents, commits, releases, tests, RAM addresses or hex "
    "numbers; call the changes today's update."
)


class ReportSummarizer:
    def __init__(self, game: str, models: tuple[str, ...], api_key: str) -> None:
        self.instructions = INSTRUCTIONS.format(game=game)
        self.models = models
        self.api_key = api_key

    def summarize(self, report: str) -> list[str]:
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "models": list(self.models),
                "max_tokens": MAX_TOKENS,
                "reasoning": {"enabled": False},
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": report},
                ],
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        content = str(response.json()["choices"][0]["message"]["content"])
        lines = [line.strip(DECORATION) for line in content.splitlines() if line.strip(DECORATION)]
        if len(lines) < SUMMARY_LINES:
            raise ValueError(f"Expected {SUMMARY_LINES} summary lines, got: {content!r}")
        return lines[:SUMMARY_LINES]


class ReportDigest:
    def __init__(self, reports_dir: Path, summarizer: ReportSummarizer) -> None:
        self.reports_dir = reports_dir
        self.summarizer = summarizer
        self._lock = threading.Lock()

    def latest(self) -> dict[str, Any]:
        reports = list_reports(self.reports_dir)
        if not reports:
            raise FileNotFoundError(self.reports_dir)
        report = self.reports_dir / str(reports[0]["name"])
        summary = report.with_suffix(SUMMARY_SUFFIX)
        with self._lock:
            if not summary.is_file() or summary.stat().st_mtime < report.stat().st_mtime:
                self._write(report, summary)
            return json.loads(summary.read_text(encoding="utf-8"))

    def _write(self, report: Path, summary: Path) -> None:
        headline, *lines = self.summarizer.summarize(read_report(self.reports_dir, report.name)["content"])
        content = json.dumps({"name": report.name, "headline": headline, "lines": lines}, indent=2)
        summary.write_text(content, encoding="utf-8")


def report_digest(settings: UISettings, game: str, reports_dir: Path) -> ReportDigest | None:
    if not settings.twitch:
        return None
    if OPENROUTER_KEY not in os.environ:
        raise RuntimeError(f"twitch.enabled is on but {OPENROUTER_KEY} is not set.")
    return ReportDigest(reports_dir, ReportSummarizer(game, settings.summary_models, os.environ[OPENROUTER_KEY]))
