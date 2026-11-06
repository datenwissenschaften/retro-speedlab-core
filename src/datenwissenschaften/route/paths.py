from pathlib import Path

from datenwissenschaften.reference.paths import reference_dir
from datenwissenschaften.settings import RetroSpeedlabConfig

ROUTE_FILE = "route.json"
VIDEO_ROUTE_FILE = "video_route.json"


def route_path(settings: RetroSpeedlabConfig) -> Path:
    return reference_dir(settings) / ROUTE_FILE


def video_route_path(settings: RetroSpeedlabConfig) -> Path:
    return reference_dir(settings) / VIDEO_ROUTE_FILE
