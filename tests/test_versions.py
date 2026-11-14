from pathlib import Path

from fakes import FAKE_VERSION, saved_state, saving_environment
from loguru import logger

from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.routes import (
    FULL_RUN,
    Leg,
    Part,
    Routes,
    load_progress,
    save_progress,
    stored_progress,
)
from datenwissenschaften.speedrun.runner import SegmentRunner, Start
from datenwissenschaften.speedrun.versions import upgrade

OLD_VERSION = "0.9.0"
PARTS = [Part("Survive", "Survive", "Boss")]
STALE = bytes([9])


def stored(tmp_path: Path, version: str, segments: list[Segment]) -> Routes:
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", PARTS, version)
    routes.save(Leg("Survive", Start("Survive", bytes([0])), "Boss"), segments)
    save_progress(routes.progress, segments, version)
    return Routes(routes.path, routes.progress, PARTS, FAKE_VERSION)


def upgraded(tmp_path: Path, version: str, segments: list[Segment]) -> tuple[Routes, list[str]]:
    routes, messages = stored(tmp_path, version, segments), []
    sink = logger.add(messages.append, format="{message}")
    upgrade(routes, SegmentRunner(saving_environment(tmp_path)))
    logger.remove(sink)
    return Routes(routes.path, routes.progress, PARTS, FAKE_VERSION), messages


def test_a_new_game_version_keeps_every_piece_that_still_replays_with_the_state_it_reaches_now(tmp_path: Path):
    pieces = [Segment("Survive", [[1, 0]], STALE, False), Segment("Boss", [[1, 0]] * 2, STALE, False)]

    kept, messages = upgraded(tmp_path, OLD_VERSION, pieces)

    route = kept.routes["Survive"]
    assert route.version == FAKE_VERSION
    assert [piece.emulator_state for piece in route.segments] == [saved_state(1), saved_state(3)]
    assert len(load_progress(kept.progress)) == 2
    assert len(messages) == 1 and f"from version {OLD_VERSION} to {FAKE_VERSION}: 4 of 4" in messages[0]


def test_a_new_game_version_trims_a_route_at_its_first_piece_that_breaks(tmp_path: Path):
    pieces = [
        Segment("Survive", [[1, 0]], STALE, False),
        Segment("Boss", [[1, 0]], STALE, False),
        Segment("Boss", [[1, 0]] * 4, STALE, True),
    ]

    kept, messages = upgraded(tmp_path, OLD_VERSION, pieces)

    assert [piece.state for piece in kept.segments("Survive")] == ["Survive"]
    assert [piece.state for piece in load_progress(kept.progress)] == ["Survive"]
    progress = stored_progress(kept.progress)
    assert progress is not None and progress.version == FAKE_VERSION
    assert f"2 of 6 stored pieces still replay (Survive 1/3; {FULL_RUN} 1/3)" in messages[0]


def test_routes_of_the_current_version_are_left_as_they_are(tmp_path: Path):
    pieces = [Segment("Boss", [[1, 0]], STALE, False)]

    kept, messages = upgraded(tmp_path, FAKE_VERSION, pieces)

    assert kept.segments("Survive") == pieces and messages == []
