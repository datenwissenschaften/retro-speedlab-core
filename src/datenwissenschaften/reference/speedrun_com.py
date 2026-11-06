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


def video_links(run: dict[str, Any]) -> list[str]:
    videos = run["videos"]
    if videos is None or "links" not in videos:
        return []
    return [link["uri"] for link in videos["links"]]


def video_url(source: SpeedrunComSource) -> str:
    url = LEADERBOARD_URL.format(game=source.game, category=source.category)
    query = {"top": 1, **{f"var-{variable}": value for variable, value in source.variables.items()}}
    response = httpx.get(url, params=query, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    links = [link for place in response.json()["data"]["runs"] for link in video_links(place["run"])]
    if not links:
        raise ValueError(f"The top speedrun.com run of {source.game}/{source.category} {source.variables} has no video")
    return links[0]


def download(url: str, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    video = directory / f"{VIDEO_FILE}.%(ext)s"
    command = [sys.executable, "-m", "yt_dlp", "--quiet", "--no-playlist", "-f", VIDEO_FORMAT, "-o", str(video), url]
    subprocess.run(command, check=True)
