import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

from datenwissenschaften.reference.paths import VIDEO_FILE
from datenwissenschaften.settings import SpeedrunComSource

LEADERBOARD_URL = "https://www.speedrun.com/api/v1/leaderboards/{game}/category/{category}"
TIMEOUT_SECONDS = 60
VIDEO_FORMAT = "best[ext=mp4]/best"
TOP_PLACES = 5


def video_links(run: dict[str, Any]) -> list[str]:
    videos = run["videos"]
    if videos is None or "links" not in videos:
        return []
    return [link["uri"] for link in videos["links"]]


def video_urls(source: SpeedrunComSource) -> list[str]:
    url = LEADERBOARD_URL.format(game=source.game, category=source.category)
    query = {"top": TOP_PLACES, **{f"var-{variable}": value for variable, value in source.variables.items()}}
    response = httpx.get(url, params=query, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return [link for place in response.json()["data"]["runs"] for link in video_links(place["run"])]


def download(url: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / f"{VIDEO_FILE}.%(ext)s"
    command = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-playlist", "-f", VIDEO_FORMAT, "-o", str(video), url]
    subprocess.run(command, check=True)
