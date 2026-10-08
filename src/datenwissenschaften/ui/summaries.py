import json
import os
import threading
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from datenwissenschaften.settings import UISettings

OPENROUTER_KEY = "OPENROUTER_API_KEY"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TIMEOUT_SECONDS = 60
MAX_TOKENS = 300
SUMMARY_LINES = 4
MIN_SUMMARY_LINES = 3
SUMMARY_ATTEMPTS = 3
RETRY_WAIT_SECONDS = 2
SUMMARY_SUFFIX = ".summary.json"
REPORT_SUFFIX = ".md"
DECORATION = " \t-*•#>\"'`"
CARD_RULES = (
    "Plain text only, without markdown, bullets, numbering, emojis or quotes. Never mention Claude, coding agents, "
    "commits, releases, tests, RAM addresses or hex numbers"
)
REPORT_INSTRUCTIONS = (
    "You write the lab update card on a live stream where Laya, an AI, teaches itself to play {game} by trial "
    "and error. Read today's lab report and write exactly four lines for the viewers. Line 1: a catchy headline "
    "of at most six words. Lines 2 to 4: one short, lively sentence each, at most 14 words: what changed today, "
    "how Laya is doing, and what to watch for next. " + CARD_RULES + "; call the changes today's update."
)


class Summarizer:
    def __init__(self, instructions: str, models: tuple[str, ...], api_key: str) -> None:
        self.instructions = instructions
        self.models = models
        self.api_key = api_key

    @retry(
        retry=retry_if_exception_type((ValueError, httpx.HTTPError)),
        stop=stop_after_attempt(SUMMARY_ATTEMPTS),
        wait=wait_exponential(multiplier=RETRY_WAIT_SECONDS),
        reraise=True,
    )
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
        if len(lines) < MIN_SUMMARY_LINES:
            raise ValueError(f"Expected {SUMMARY_LINES} summary lines, got: {content!r}")
        return lines[:SUMMARY_LINES]


class CachedSummary:
    def __init__(self, summarizer: Summarizer) -> None:
        self.summarizer = summarizer
        self._lock = threading.Lock()

    def of(self, source: Path) -> dict[str, Any]:
        if not source.is_file():
            raise FileNotFoundError(source)
        summary = source.with_suffix(SUMMARY_SUFFIX)
        with self._lock:
            if not summary.is_file() or summary.stat().st_mtime < source.stat().st_mtime:
                headline, *lines = self.summarizer.summarize(source.read_text(encoding="utf-8"))
                content = {"name": source.name, "headline": headline, "lines": lines}
                summary.write_text(json.dumps(content, indent=2), encoding="utf-8")
            return json.loads(summary.read_text(encoding="utf-8"))


class ReportDigest:
    def __init__(self, reports_dir: Path, summary: CachedSummary) -> None:
        self.reports_dir = reports_dir
        self.summary = summary

    def latest(self) -> dict[str, Any]:
        reports = sorted(self.reports_dir.glob(f"*{REPORT_SUFFIX}"), key=lambda report: report.name)
        if not reports:
            raise FileNotFoundError(self.reports_dir)
        return self.summary.of(reports[-1])


def report_digest(settings: UISettings, game: str, reports_dir: Path) -> ReportDigest | None:
    if not settings.twitch:
        return None
    if OPENROUTER_KEY not in os.environ:
        raise RuntimeError(f"twitch.enabled is on but {OPENROUTER_KEY} is not set.")
    summarizer = Summarizer(REPORT_INSTRUCTIONS.format(game=game), settings.summary_models, os.environ[OPENROUTER_KEY])
    return ReportDigest(reports_dir, CachedSummary(summarizer))
