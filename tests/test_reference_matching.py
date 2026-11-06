import json
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import msgspec
import numpy as np
import pytest
from fakes import FakeRam, FakeWrapper, SavingEmulator, fake_routes, write_config

from datenwissenschaften.reference import frames, matching, screens
from datenwissenschaften.reference.explored import archived_starts
from datenwissenschaften.reference.matcher import MATCH_DISTANCE, match_screens
from datenwissenschaften.reference.thumbnail import THUMB_SHAPE, content_box
from datenwissenschaften.route.position import Position
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.archive_store import StoredArchive, StoredCell
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

SCRIPT = [(3, 0), (3, 1), (5, 2), (3, 3), (3, 4)]
SCREEN = (60, 64)
VIDEO = (120, 192)
PILLAR = 32
PLACE_PIXELS = 64
DARK_FRAMES = 16


def pattern(seed: int, shape: tuple[int, int]) -> np.ndarray:
    return np.random.default_rng(seed).integers(40, 256, (*shape, 3), dtype=np.uint8)


def thumbs(seeds: list[int]) -> np.ndarray:
    return np.stack([pattern(seed, THUMB_SHAPE[:2]) for seed in seeds])


def cells(places: list[tuple[int, int, int]]) -> np.ndarray:
    return np.asarray(places, dtype=np.int64)


def test_a_video_frame_matches_the_emulator_screen_it_shows_whatever_its_brightness_and_contrast():
    emulator = thumbs([0, 1, 2])
    video = (emulator[[2, 0]] * 0.6 + 30).astype(np.uint8)

    found = match_screens(video, emulator, cells([(0, 0, 0), (0, 5, 0), (0, 10, 0)]))

    assert [(hit.frame, hit.screen) for hit in found] == [(0, 2), (1, 0)]
    assert all(hit.distance < MATCH_DISTANCE for hit in found)


def test_a_frame_of_a_place_no_emulator_screen_shows_stays_unmatched():
    assert match_screens(thumbs([7]), thumbs([0, 1]), cells([(0, 0, 0), (0, 5, 0)])) == []


def test_a_frame_two_distinct_places_show_alike_stays_unmatched_but_neighbouring_cells_do_not_compete():
    emulator = thumbs([0, 0, 1])
    video = thumbs([0])

    assert match_screens(video, emulator, cells([(0, 0, 0), (0, 9, 0), (0, 20, 0)])) == []
    assert len(match_screens(video, emulator, cells([(0, 0, 0), (0, 1, 1), (0, 20, 0)]))) == 1


def test_a_level_known_at_one_place_only_matches_nothing():
    assert match_screens(thumbs([0]), thumbs([0, 0]), cells([(0, 0, 0), (0, 1, 0)])) == []


def test_the_hud_rows_that_look_alike_everywhere_do_not_decide_the_match():
    emulator = thumbs([0, 1, 2])
    emulator[:, :4] = 200
    video = emulator[[1]].copy()
    video[:, :4] = pattern(9, (4, THUMB_SHAPE[1]))

    found = match_screens(video, emulator, cells([(0, 0, 0), (0, 5, 0), (0, 10, 0)]))

    assert [hit.screen for hit in found] == [1]


def test_nothing_to_compare_matches_nothing():
    assert match_screens(thumbs([0]), np.zeros((0, *THUMB_SHAPE), np.uint8), cells([]).reshape(0, 3)) == []


def test_the_black_border_around_the_game_is_cut_away_and_an_all_black_video_fails():
    frame = np.zeros((10, 20, 3), np.uint8)
    frame[2:8, 5:15] = 100

    assert content_box(frame) == (slice(2, 8), slice(5, 15))
    with pytest.raises(ValueError, match="black"):
        content_box(np.zeros((4, 4, 3), np.uint8))


def video_frame(seed: int | None) -> np.ndarray:
    frame = np.zeros((*VIDEO, 3), np.uint8)
    if seed is not None:
        game = cv2.resize(pattern(seed, SCREEN), (VIDEO[1] - 2 * PILLAR, VIDEO[0]), interpolation=cv2.INTER_NEAREST)
        frame[:, PILLAR:-PILLAR] = (game * 0.8 + 20).astype(np.uint8)
    return frame


def write_frames(directory: Path, seeds: list[int | None]) -> None:
    directory.mkdir(parents=True)
    for index, seed in enumerate(seeds):
        cv2.imwrite(str(directory / f"{index / 4:07.2f}s.png"), cv2.cvtColor(video_frame(seed), cv2.COLOR_RGB2BGR))


def test_video_frames_become_cached_thumbnails_of_the_game_inside_its_border(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    write_frames(tmp_path / "frames", [1, None, 2])

    read = frames.load_frames(tmp_path / "frames", tmp_path / "frames.npz")
    monkeypatch.setattr(frames, "read_frames", lambda files: pytest.fail("the cache is read"))
    cached = frames.load_frames(tmp_path / "frames", tmp_path / "frames.npz")

    assert read.seconds.tolist() == [0.0, 0.25, 0.5] and read.thumbs.shape == (3, *THUMB_SHAPE)
    assert read.thumbs[0].min() >= 40 and read.thumbs[1].max() == 0
    assert np.array_equal(cached.thumbs, read.thumbs)


def test_a_reference_without_frames_or_with_a_broken_frame_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        frames.read_frames([])
    (tmp_path / "0000.00s.png").write_bytes(b"no png")
    with pytest.raises(ValueError, match="no image"):
        frames.read_frames([tmp_path / "0000.00s.png"])


@dataclass
class WalkingRam(FakeRam):
    def position(self) -> Position | None:
        return None if self.score == 0 else Position(0, self.score * PLACE_PIXELS, 0, 0, False)

    def dynamics(self) -> tuple[int, ...]:
        return (self.lives,)


class WalkingWrapper(FakeWrapper):
    ram_info_cls = WalkingRam


def walking_runner(tmp_path: Path) -> SegmentRunner:
    emulator = SavingEmulator(tmp_path / "recordings", SCRIPT)
    emulator.em.get_screen = lambda: pattern(emulator.position, SCREEN)
    return SegmentRunner(WalkingWrapper(emulator, fake_routes(tmp_path)))


def stored_cell(key: list[object], state: int) -> StoredCell:
    return StoredCell(key, bytes([state]), None, 0, 0, 0, None, 0.0, 0)


def archive(directory: Path, positions: list[int]) -> None:
    placed = [stored_cell(["Survive", 0, place, 0, 0, False, []], place) for place in positions]
    again = stored_cell(["Survive", 0, positions[0], 0, 0, False, []], 9)
    unplaced = stored_cell(["Survive", 7], 8)
    (directory / "key").mkdir(parents=True, exist_ok=True)
    stored = StoredArchive("Survive", b"", 0, [*placed, again, unplaced])
    (directory / "key" / "1.msgpack").write_bytes(msgspec.msgpack.encode(stored))


def test_every_placed_archive_cell_is_one_start_once(tmp_path: Path):
    archive(tmp_path, [0, 1])

    assert archived_starts(tmp_path) == [Start("Survive", bytes([0])), Start("Survive", bytes([1]))]


def test_archived_places_are_rendered_once_with_their_position_level_and_dynamics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive(tmp_path / "archives", [0, 1, 4])
    runner = walking_runner(tmp_path)

    shots = screens.updated_screens(runner, tmp_path / "archives", tmp_path / "screens.msgpack")
    monkeypatch.setattr(screens, "screenshot", lambda runner, start: pytest.fail("rendered again"))
    again = screens.updated_screens(runner, tmp_path / "archives", tmp_path / "screens.msgpack")

    assert [(shot.x, shot.level, shot.dynamics) for shot in shots] == [(64, 0, [3]), (128, 0, [5]), (256, 0, [3])]
    assert np.array_equal(shots[0].image(), cv2.resize(pattern(1, SCREEN), (32, 30), interpolation=cv2.INTER_AREA))
    assert again == shots


def test_a_place_without_position_after_its_frame_gives_no_screen(tmp_path: Path):
    runner = walking_runner(tmp_path)
    runner.emulator.script = [(3, 0), (3, 0)]

    assert screens.screenshot(runner, Start("Survive", bytes([0]))) is None


class NowPool:
    def __init__(self) -> None:
        self.jobs = 0

    def submit(self, job, *args) -> Future:
        self.jobs += 1
        done: Future = Future()
        try:
            done.set_result(job(*args))
        except RuntimeError as error:
            done.set_exception(error)
        return done


def reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeds: list[int | None]) -> Path:
    config = write_config(tmp_path)
    cache = load_config(config).paths.cache_dir
    write_frames(cache / "reference" / "frames", seeds)
    archive(cache / "tas" / "archives", [0, 1, 2, 3])
    runner = walking_runner(tmp_path)
    monkeypatch.setattr(matching, "worker_runner", lambda wrapper, config: runner)
    return config


def level_video() -> list[int | None]:
    return [*[None] * DARK_FRAMES, 1, 1, 7, 2, 2, 1, *[None] * DARK_FRAMES]


def test_the_reference_video_is_matched_with_emulator_screens_into_waypoints_a_route_and_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = reference(tmp_path, monkeypatch, level_video())
    reported: list[list[dict[str, int]]] = []
    pool = NowPool()
    referencing = matching.ReferenceMatching("Fake", config, pool, SimpleNamespace(referenced=reported.append))

    referencing.refresh()

    folder = load_config(config).paths.cache_dir / "reference"
    found = json.loads((folder / "matches.json").read_text(encoding="utf-8"))
    assert reported == [[{"level": 0, "matched": 5, "frames": 6}]]
    assert [(match["seconds"], match["x"], match["cell"]) for match in found["matches"]] == [
        (4.0, 64, [0, 4, 0]),
        (4.25, 64, [0, 4, 0]),
        (4.75, 128, [0, 8, 0]),
        (5.0, 128, [0, 8, 0]),
        (5.25, 64, [0, 4, 0]),
    ]
    assert found["matches"][0]["state"] == "Survive" and found["matches"][0]["level"] == 0
    assert found["levels"][0]["waypoints"] == [
        {"seconds": 4.0, "cell": [0, 4, 0], "state": "Survive"},
        {"seconds": 4.75, "cell": [0, 8, 0], "state": "Survive"},
        {"seconds": 5.25, "cell": [0, 4, 0], "state": "Survive"},
    ]
    route = json.loads((folder / "video_route.json").read_text(encoding="utf-8"))
    assert route == [[0, 4, 0, 315, 15, 3], [0, 8, 0, 300, 30, 5]]
    routes = RouteMap((folder / "video_route.json",))
    assert routes.remaining(Position(0, 130, 0, 0, False), (5,)) == 30
    assert routes.remaining(Position(0, 130, 0, 0, False), (3,)) is None


def test_matching_waits_for_frames_and_for_the_matching_before(tmp_path: Path):
    config = write_config(tmp_path)
    pool = NowPool()
    referencing = matching.ReferenceMatching("Fake", config, pool, SimpleNamespace(referenced=[].append))

    referencing.refresh()
    write_frames(load_config(config).paths.cache_dir / "reference" / "frames", [None])
    referencing.running = Future()
    referencing.refresh()

    assert pool.jobs == 0


def test_a_video_whose_levels_do_not_split_logs_the_failure_and_reports_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = reference(tmp_path, monkeypatch, [1, 2])
    reported: list[list[dict[str, int]]] = []
    referencing = matching.ReferenceMatching("Fake", config, NowPool(), SimpleNamespace(referenced=reported.append))

    referencing.refresh()

    assert reported == [] and referencing.running is not None and referencing.running.exception() is not None
