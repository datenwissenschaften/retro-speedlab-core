import io
import zipfile
from pathlib import Path

import httpx

PUBLICATION_URL = "https://tasvideos.org/api/v1/publications/{publication}"
MOVIE_URL = "https://tasvideos.org/{publication}M?handler=Download"
VIDEO_SUFFIXES = (".mkv", ".mp4")
TIMEOUT_SECONDS = 600


def video_url(publication: int) -> str:
    response = httpx.get(PUBLICATION_URL.format(publication=publication), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    urls = response.json()["urls"]
    videos = [url for url in urls if url.endswith(VIDEO_SUFFIXES)]
    if not videos:
        raise ValueError(f"TASVideos publication {publication} links no {VIDEO_SUFFIXES} video: {urls}")
    return videos[0]


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)


def save_movie(publication: int, directory: Path) -> None:
    response = httpx.get(MOVIE_URL.format(publication=publication), follow_redirects=True, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(directory)
