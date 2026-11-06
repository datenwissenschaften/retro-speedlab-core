from pathlib import Path

from datenwissenschaften.reference.paths import reference_dir
from datenwissenschaften.settings import RetroSpeedlabConfig

ROUTE_FILE = "route.json"
RAM_ROUTE_FILE = "tas_ram_route.json"


def route_path(settings: RetroSpeedlabConfig) -> Path:
    return reference_dir(settings) / ROUTE_FILE


def ram_route_path(settings: RetroSpeedlabConfig) -> Path:
    return settings.paths.game_dir / RAM_ROUTE_FILE


def route_paths(settings: RetroSpeedlabConfig) -> tuple[Path, ...]:
    return route_path(settings), ram_route_path(settings)
