import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui.reports import list_reports

OPENROUTER_KEY = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60
MAX_TOKENS = 300
SUMMARY_LINES = 4
SUMMARY_SUFFIX = ".summary.json"
DECORATION = " \t-*•#>\"'`"
CARD_FORMAT = (
    "Write exactly four lines for the viewers. Line 1: a catchy headline of at most six words. Lines 2 to 4: one "
    "short, lively sentence each, at most 14 words. Plain text only, without markdown, bullets, numbering, emojis "
    "or quotes. Never mention Claude, coding agents, commits, releases, tests, RAM addresses or hex numbers."
)
REPORT_INSTRUCTIONS = (
    "You write the lab update card on a live stream where Laya, an AI, teaches itself to play {game} by trial "
    "and error. Read today's lab report. " + CARD_FORMAT + " Lines 2 to 4 say what changed today, how Laya is "
    "doing, and what to watch for next; call the changes today's update."
)
HINT_INSTRUCTIONS = (
    "You write the developer hints card on a live stream where Laya, an AI, teaches itself to play {game} by "
    "trial and error. Read the hints the developer gave the lab about the game. " + CARD_FORMAT + " Lines 2 to 4 "
    "say in plain game terms what the developer told the lab, for example what wins a level or what to avoid."
)


class Summarizer:
    def __init__(self, instructions: str, models: tuple[str, ...], api_key: str) -> None:
        self.instructions = instructions
        self.models = models
        self.api_key = api_key

    def summarize(self, text: str) -> list[str]:
        response = httpx.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "models": list(self.models),
                "max_tokens": MAX_TOKENS,
                "reasoning": {"enabled": False},
                "messages": [
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": text},
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


def summary_path(source: Path) -> Path:
    return source.with_suffix(SUMMARY_SUFFIX)


class CachedSummary:
    def __init__(self, summarizer: Summarizer) -> None:
        self.summarizer = summarizer
        self._lock = threading.Lock()

    def of(self, source: Path) -> dict[str, Any]:
        if not source.is_file():
            raise FileNotFoundError(source)
        summary = summary_path(source)
        with self._lock:
            if not summary.is_file() or summary.stat().st_mtime < source.stat().st_mtime:
                headline, *lines = self.summarizer.summarize(source.read_text(encoding="utf-8"))
                content = {"name": source.name, "headline": headline, "lines": lines}
                summary.write_text(json.dumps(content, indent=2), encoding="utf-8")
            return json.loads(summary.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class LabSummaries:
    reports_dir: Path
    hints_file: Path
    reports: CachedSummary
    hints: CachedSummary

    def latest_report(self) -> dict[str, Any]:
        reports = list_reports(self.reports_dir)
        if not reports:
            raise FileNotFoundError(self.reports_dir)
        return self.reports.of(self.reports_dir / str(reports[0]["name"]))

    def hint(self) -> dict[str, Any]:
        return self.hints.of(self.hints_file)


def lab_summaries(settings: UISettings, game: str, reports_dir: Path, hints_file: Path) -> LabSummaries | None:
    if not settings.twitch:
        return None
    if OPENROUTER_KEY not in os.environ:
        raise RuntimeError(f"twitch.enabled is on but {OPENROUTER_KEY} is not set.")
    api_key = os.environ[OPENROUTER_KEY]
    return LabSummaries(
        reports_dir=reports_dir,
        hints_file=hints_file,
        reports=CachedSummary(Summarizer(REPORT_INSTRUCTIONS.format(game=game), settings.summary_models, api_key)),
        hints=CachedSummary(Summarizer(HINT_INSTRUCTIONS.format(game=game), settings.summary_models, api_key)),
    )
