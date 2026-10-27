import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

SHOWN_COMMITS = 5
SUBJECT_CHARACTERS = 100
UNPRINTABLE = re.compile(r"[^\x20-\x7e]")
SECRET_LIKE = re.compile(r"\b(?=[A-Za-z0-9_+/=-]*\d)(?=[A-Za-z0-9_+/=-]*[A-Za-z])[A-Za-z0-9_+/=-]{20,}\b")
URL = re.compile(r"\S+://\S+")
HIDDEN = "[hidden]"
BINARY = "-"


@dataclass(slots=True, frozen=True)
class BranchChanges:
    commits: int
    subjects: list[str]
    files: int
    added: int
    removed: int


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
