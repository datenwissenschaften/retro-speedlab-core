import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import msgspec
import numpy as np
import pytest
from fakes import FakeRam, FakeWrapper, SavingEmulator, fake_routes

from datenwissenschaften.reference import play, samples, video
from datenwissenschaften.reference.explored import STARTS_PER_CELL, explored_starts
from datenwissenschaften.route.position import Position
from datenwissenschaften.speedrun.archive_store import StoredArchive, StoredCell
from datenwissenschaften.speedrun.movie import Segment
from datenwissenschaften.speedrun.runner import SegmentRunner, Start

SCRIPT = [(3, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
LEVELLED_SCRIPT = [(3, 0), (3, 1), (3, 2), (4, 3), (4, 4), (4, 5)]
AREAS_PER_LEVEL = 256
SCREEN = (32, 48)


@dataclass
class PlacedRam(FakeRam):
    def position(self) -> Position | None:
        return None if self.score == 0 else Position(0, self.score * 16, self.lives * 16, 0, False)


class PlacedWrapper(FakeWrapper):
    ram_info_cls = PlacedRam


def placed_runner(tmp_path: Path) -> SegmentRunner:
    emulator = SavingEmulator(tmp_path / "recordings", SCRIPT)
    emulator.em.get_screen = lambda: np.full((*SCREEN, 3), emulator.position * 40, np.uint8)
    return SegmentRunner(PlacedWrapper(emulator, fake_routes(tmp_path)))


@dataclass
class LevelledRam(FakeRam):
    def position(self) -> Position | None:
        return None if self.score == 0 else Position(self.level() * AREAS_PER_LEVEL, self.score, 0, 0, False)

    def level(self) -> int:
        return self.lives - 3


class LevelledWrapper(FakeWrapper):
    ram_info_cls = LevelledRam


def levelled_runner(tmp_path: Path) -> SegmentRunner:
    return SegmentRunner(
        LevelledWrapper(SavingEmulator(tmp_path / "recordings", LEVELLED_SCRIPT), fake_routes(tmp_path))
    )


def route_taken() -> list[Segment]:
    return [Segment("Survive", [[1, 0]] * 4, b"", False)]


def test_samples_label_every_screen_with_its_place_and_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 20)
    runner = placed_runner(tmp_path)

    collected = samples.collect_samples(runner, [(runner.power_on(), route_taken())], [], [], 0)

    assert len(collected.labels) >= 20 and len(collected.screens) == len(collected.labels)
    assert collected.screens[0].shape == (SCREEN[0] // 2, SCREEN[1] // 2, 3)
    assert {label[3] for label in collected.labels} == {0}
    assert collected.labels[0] == (0, 32, 48, 0)


def test_samples_of_a_later_level_start_where_the_loader_brought_it_and_carry_that_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 20)
    runner = levelled_runner(tmp_path)

    collected = samples.collect_samples(
        runner, [(runner.power_on(), route_taken())], [Start("Survive", bytes([3]))], [], 0
    )

    placed = [label for label in collected.labels if label[0] >= 0]
    assert sum(label[3] == 1 for label in placed) >= 20
    assert all(label[0] == label[3] * AREAS_PER_LEVEL for label in placed)
    assert collected.labels[-1] == (AREAS_PER_LEVEL, 5, 0, 1)


def test_places_the_search_explored_add_screens_of_their_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 1)

    collected = samples.collect_samples(levelled_runner(tmp_path), [], [], [Start("Survive", bytes([4]))], 0)

    assert (AREAS_PER_LEVEL, 5, 0, 1) in collected.labels


def stored_cell(key: list[object], state: int) -> StoredCell:
    return StoredCell(key, bytes([state]), None, 0, 0, 0, None, 0.0, 0)


def test_a_few_archived_cells_per_route_cell_are_kept_as_starts(tmp_path: Path):
    near = [stored_cell(["Boss", 3, 4, 6, 0, False, []], state) for state in range(3)]
    other = stored_cell(["Boss", 3, 40, 6, 0, False, []], 7)
    unplaced = stored_cell(["Boss", 9], 8)
    (tmp_path / "key").mkdir()
    archive = StoredArchive("Boss", b"", 0, [*near, other, unplaced])
    (tmp_path / "key" / "1.msgpack").write_bytes(msgspec.msgpack.encode(archive))

    starts = explored_starts(tmp_path, random.Random(0))

    assert len(starts) == STARTS_PER_CELL + 1
    assert {start.state for start in starts} == {"Boss"}
    assert Start("Boss", bytes([7])) in starts and Start("Boss", bytes([8])) not in starts


def test_a_video_is_read_in_chunks_of_small_screens(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(video, "CHUNK_FRAMES", 2)
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (64, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 64, 3), np.uint8))
    writer.release()

    chunks = list(video.video_chunks(path))

    assert [chunk.shape for chunk in chunks] == [(2, 16, 24, 3), (1, 16, 24, 3)]


def test_the_nearest_distance_of_each_query_to_the_seen_screens():
    torch = pytest.importorskip("torch")
    from datenwissenschaften.reference.familiar import nearest

    seen = torch.tensor([[0.0, 0.0], [10.0, 0.0]])

    assert nearest(torch.tensor([[0.0, 3.0], [9.0, 0.0]]), seen).tolist() == [3.0, 1.0]


def test_a_level_is_the_play_between_two_long_dark_screens_and_short_fades_stay_inside_it():
    dark = np.concatenate(
        [np.ones(200), np.zeros(300), np.ones(30), np.zeros(300), np.ones(200), np.zeros(400), np.ones(200)]
    )

    assert play.level_spans(dark.astype(bool), 2, 60) == [(200, 830), (1030, 1430)]


def test_a_score_screen_split_by_a_flash_is_one_break():
    dark = np.concatenate(
        [np.ones(200), np.zeros(600), np.ones(100), np.zeros(10), np.ones(100), np.zeros(600), np.ones(200)]
    )

    assert play.level_spans(dark.astype(bool), 2, 60) == [(200, 800), (1010, 1610)]


def test_a_video_whose_dark_screens_frame_other_levels_than_the_game_has_fails():
    dark = np.concatenate([np.zeros(300), np.ones(200), np.zeros(300)]).astype(bool)

    with pytest.raises(RuntimeError, match="0 levels"):
        play.level_spans(dark, 1, 60)


def test_a_score_screen_is_dark_but_a_dim_level_is_not():
    score = np.zeros((1, 32, 32, 3), np.uint8)
    score[0, :4] = 255
    dim = np.full((1, 32, 32, 3), (0, 0, 120), np.uint8)

    assert play.dark_screens(np.concatenate([score, dim])).tolist() == [True, False]
