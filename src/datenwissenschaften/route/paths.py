from pathlib import Path

from datenwissenschaften.settings import RetroSpeedlabConfig

ROUTE_FILE = "route.json"


def route_path(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.cache_dir / ROUTE_FILE
