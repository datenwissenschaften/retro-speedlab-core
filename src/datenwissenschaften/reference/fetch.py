import subprocess
from pathlib import Path

from loguru import logger

from datenwissenschaften.reference.paths import FRAMES_DIR, MOVIE_DIR, VIDEO_FILE, reference_dir
from datenwissenschaften.reference.tasvideos import download, save_movie, video_url
from datenwissenschaften.settings import RetroSpeedlabConfig

FRAMES_PER_SECOND = 4


def extract_frames(video: Path, frames: Path) -> None:
    frames.mkdir(parents=True, exist_ok=True)
    pattern = frames / "%05d.png"
    command = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"fps={FRAMES_PER_SECOND}", str(pattern)]
    subprocess.run(command, check=True)
    for index, frame in enumerate(sorted(frames.glob("*.png"))):
        frame.rename(frames / f"{index / FRAMES_PER_SECOND:07.2f}s.png")


def ensure_reference(settings: RetroSpeedlabConfig) -> None:
    directory = reference_dir(settings)
    frames = directory / FRAMES_DIR
    if frames.is_dir():
        return
    publication = settings.speedrun.game.tasvideos
    url = video_url(publication)
    video = directory / f"{VIDEO_FILE}{Path(url).suffix}"
    download(url, video)
    save_movie(publication, directory / MOVIE_DIR)
    extract_frames(video, directory / f".{FRAMES_DIR}")
    (directory / f".{FRAMES_DIR}").rename(frames)
    logger.info(f"TASVideos {publication} is the lab's reference: {len(list(frames.glob('*.png')))} frames in {frames}")
