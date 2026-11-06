from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

REFERENCE_DIR = "reference"
FRAMES_DIR = "frames"
VIDEO_FILE = "video"
MATCHES_FILE = "matches.json"
SCREENS_FILE = "screens.msgpack"
THUMBS_FILE = "frames.npz"


def reference_dir(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / REFERENCE_DIR


def video_path(settings: RetroSpeedlabConfig) -> Path:
    videos = sorted(reference_dir(settings).glob(f"{VIDEO_FILE}.*"))
    if len(videos) != 1:
        raise FileNotFoundError(f"The reference holds {videos}, not one {VIDEO_FILE} file")
    return videos[0]
