import subprocess
from pathlib import Path

from loguru import logger

from datenwissenschaften.reference.paths import FRAMES_DIR, reference_dir, video_path
from datenwissenschaften.reference.speedrun_com import download, video_urls
from datenwissenschaften.settings import RetroSpeedlabConfig

FRAMES_PER_SECOND = 4


def extract_frames(video: Path, frames: Path) -> None:
    frames.mkdir(parents=True, exist_ok=True)
    pattern = frames / "%05d.png"
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FRAMES_PER_SECOND}", str(pattern)]
    subprocess.run(command, check=True)
    for index, frame in enumerate(sorted(frames.glob("*.png"))):
        frame.rename(frames / f"{index / FRAMES_PER_SECOND:07.2f}s.png")


def downloaded(urls: list[str], directory: Path) -> str | None:
    for url in urls:
        try:
            download(url, directory)
        except subprocess.CalledProcessError:
            logger.warning(f"The speedrun.com video {url} does not download")
            continue
        return url
    return None


def ensure_reference(settings: RetroSpeedlabConfig) -> None:
    directory = reference_dir(settings)
    frames = directory / FRAMES_DIR
    if frames.is_dir():
        return
    url = downloaded(video_urls(settings.speedrun.game.speedrun_com), directory)
    if url is None:
        logger.warning("No video of the top speedrun.com runs downloads; the lab goes on without a reference")
        return
    extract_frames(video_path(settings), directory / f".{FRAMES_DIR}")
    (directory / f".{FRAMES_DIR}").rename(frames)
    logger.info(f"The speedrun.com run {url} is the lab's reference: {len(list(frames.glob('*.png')))} frames")
