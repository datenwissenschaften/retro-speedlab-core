from pathlib import Path

from loguru import logger

from datenwissenschaften.knowledge import knowledge_path
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.speedrun.hints import HINTS_FILE
from datenwissenschaften.speedrun.paths import PROGRESS_FILE, tas_dir
from datenwissenschaften.speedrun.record import RECORD_FILE, stored_record
from datenwissenschaften.speedrun.routes import (
    FULL_RUN,
    ROUTES_FILE,
    Leg,
    Routes,
    finished,
    save_progress,
    still_replaying,
    stored_progress,
    stored_routes,
)
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

Kept = tuple[str, int, int]
RETIRED_DIR = "retired"
SEMVER_PARTS = 3


def major(version: str) -> int:
    parts = version.split(".")
    if len(parts) != SEMVER_PARTS or not all(part.isdigit() for part in parts):
        raise ValueError(f"The game version {version!r} is not MAJOR.MINOR.PATCH")
    return int(parts[0])


def learned_versions(output: Path) -> set[str]:
    record = stored_record(output / RECORD_FILE)
    progress = stored_progress(output / PROGRESS_FILE)
    return {
        *(route.version for route in stored_routes(output / ROUTES_FILE).values()),
        *([] if record is None else [record.version]),
        *([] if progress is None else [progress.version]),
    }


def learned_state(settings: RetroSpeedlabConfig) -> list[Path]:
    paths = settings.paths
    return [paths.cache_dir, paths.models_dir, paths.game_dir / HINTS_FILE, knowledge_path(paths.game_dir)]


def retire_other_major(settings: RetroSpeedlabConfig) -> None:
    game = settings.speedrun.game
    older = sorted(version for version in learned_versions(tas_dir(settings)) if major(version) != major(game.version))
    if not older:
        return
    retired = settings.paths.game_dir / RETIRED_DIR / "-".join(older)
    retired.mkdir(parents=True)
    for path in learned_state(settings):
        if path.exists():
            path.replace(retired / path.name)
    logger.warning(
        f"{game.title} is version {game.version}, a new major after {', '.join(older)}: its record, routes,"
        f" progress, policy, archive, hints, knowledge and caches are retired to {retired} and the search starts fresh"
    )


def report_upgrade(version: str, found_on: set[str], counts: list[Kept]) -> None:
    kept, total = sum(count[1] for count in counts), sum(count[2] for count in counts)
    parts = "; ".join(f"{name} {replaying}/{stored}" for name, replaying, stored in counts)
    logger.info(
        f"The game changed from version {', '.join(sorted(found_on))} to {version}:"
        f" {kept} of {total} stored pieces still replay ({parts}); the search goes on from the first that broke"
    )


def upgrade(routes: Routes, runner: SegmentRunner) -> None:
    power_on = runner.power_on()
    found_on: set[str] = set()
    counts: list[Kept] = []
    end: bytes | None = power_on.emulator_state
    for part in routes.parts:
        if part.name not in routes.routes:
            end = None
            continue
        route = routes.routes[part.name]
        segments = list(route.segments)
        if route.version != routes.version:
            entry = Start(part.entry_state, route.entry if end is None else end)
            segments = still_replaying(runner, entry, segments)
            found_on.add(route.version)
            counts.append((part.name, len(segments), len(route.segments)))
            routes.save(Leg(part.name, entry, part.exit), segments)
        end = segments[-1].emulator_state if finished(segments, part.exit) else None
    progress = stored_progress(routes.progress)
    if progress is not None and progress.version != routes.version:
        kept = still_replaying(runner, power_on, list(progress.segments))
        found_on.add(progress.version)
        counts.append((FULL_RUN, len(kept), len(progress.segments)))
        save_progress(routes.progress, kept, routes.version)
    if counts:
        report_upgrade(routes.version, found_on, counts)
