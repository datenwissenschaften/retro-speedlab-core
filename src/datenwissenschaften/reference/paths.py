from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

REFERENCE_DIR = "reference"
FRAMES_DIR = "frames"
MOVIE_DIR = "movie"
VIDEO_FILE = "video"
ROUTE_FILE = "route.json"
TRACE_FILE = "trace.json"


def reference_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / REFERENCE_DIR


def route_path(settings: RetroSpeedlabConfig) -> Path:
    return reference_dir(settings) / ROUTE_FILE


def video_path(settings: RetroSpeedlabConfig) -> Path:
    videos = sorted(reference_dir(settings).glob(f"{VIDEO_FILE}.*"))
    if len(videos) != 1:
        raise FileNotFoundError(f"The reference holds {videos}, not one {VIDEO_FILE} file")
    return videos[0]
