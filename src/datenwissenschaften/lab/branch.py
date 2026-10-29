import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from datenwissenschaften.lab.activity import shown_file

SHOWN_COMMITS = 5
SUBJECT_CHARACTERS = 100
UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
SECRET_LIKE = re.compile(r"\b(?=[A-Za-z0-9_+/=-]*\d)(?=[A-Za-z0-9_+/=-]*[A-Za-z])[A-Za-z0-9_+/=-]{20,}\b")
URL = re.compile(r"\S+://\S+")
HIDDEN = "[hidden]"
BINARY = "-"
HUNK_LINES = 12
PYTHON_FILES = "*.py"
FILE_HEADER = "diff --git "
NEW_FILE = "+++ b/"
HUNK_HEADER = "@@"
CHANGED = ("+", "-")


@dataclass(slots=True, frozen=True)
class BranchChanges:
    commits: int
    subjects: list[str]
    files: int
    added: int
    removed: int


@dataclass(slots=True, frozen=True)
class CodeHunk:
    file: str
    lines: list[str]
    commit: str


def shown_subject(subject: str) -> str:
    cleaned = SECRET_LIKE.sub(HIDDEN, URL.sub(HIDDEN, UNPRINTABLE.sub("", subject)))
    return cleaned[:SUBJECT_CHARACTERS]


def git(worktree: Path, *arguments: str) -> list[str]:
    command = ["git", "-c", "safe.directory=*", "-C", str(worktree), *arguments]
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout.splitlines()


def line_counts(numstat: list[str]) -> tuple[int, int]:
    counts = [line.split("\t")[:2] for line in numstat]
    numbers = [(int(added), int(removed)) for added, removed in counts if BINARY not in (added, removed)]
    return sum(added for added, _ in numbers), sum(removed for _, removed in numbers)


def branch_changes(worktree: Path, base: str) -> BranchChanges | None:
    if not worktree.is_dir():
        return None
    try:
        subjects = git(worktree, "log", "--format=%s", f"{base}..HEAD")
        numstat = git(worktree, "diff", "--numstat", base)
    except subprocess.CalledProcessError as error:
        logger.warning(f"The lab branch {worktree.name} cannot be read: {error.stderr.strip()}")
        return None
    added, removed = line_counts(numstat)
    shown = [shown_subject(subject) for subject in subjects[:SHOWN_COMMITS]]
    return BranchChanges(len(subjects), shown, len(numstat), added, removed)


def first_hunk(diff: list[str], worktree: Path) -> tuple[str, list[str]] | None:
    file, lines = None, []
    for line in diff:
        if lines and line.startswith((FILE_HEADER, HUNK_HEADER)):
            break
        if line.startswith(FILE_HEADER):
            file = None
        elif line.startswith(NEW_FILE):
            file = shown_file(line.removeprefix(NEW_FILE), worktree)
        elif file is not None and line.startswith(CHANGED):
            lines.append(shown_subject(line))
            if len(lines) == HUNK_LINES:
                break
    return None if file is None or not lines else (file, lines)


def newest_hunk(worktree: Path, base: str) -> CodeHunk | None:
    if not worktree.is_dir():
        return None
    try:
        subjects = git(worktree, "log", "-1", "--format=%s", f"{base}..HEAD")
        diff = (
            git(worktree, "show", "--format=", "--unified=0", "--no-color", "HEAD", "--", PYTHON_FILES)
            if subjects
            else []
        )
    except subprocess.CalledProcessError as error:
        logger.warning(f"The newest lab commit in {worktree.name} cannot be read: {error.stderr.strip()}")
        return None
    hunk = first_hunk(diff, worktree)
    return None if hunk is None else CodeHunk(*hunk, shown_subject(subjects[0]))
