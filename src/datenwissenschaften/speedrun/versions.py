from loguru import logger

from datenwissenschaften.speedrun.routes import (
    FULL_RUN,
    Leg,
    Routes,
    finished,
    save_progress,
    still_replaying,
    stored_progress,
)
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

Kept = tuple[str, int, int]


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
