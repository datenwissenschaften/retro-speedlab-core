import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import msgspec
import numpy as np
import pytest
from fakes import FakeRam, FakeWrapper, SavingEmulator, fake_routes, write_config

from datenwissenschaften.reference import guide, play, samples
from datenwissenschaften.reference.explored import STARTS_PER_CELL, explored_starts
from datenwissenschaften.reference.paths import TRACE_FILE, reference_dir, route_path, tas_route_path
from datenwissenschaften.route.position import Position
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.archive_store import StoredArchive, StoredCell
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.routes import Leg, Part, Routes
from datenwissenschaften.tas.runner import SegmentRunner, Start

SCRIPT = [(3, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
LEVELLED_SCRIPT = [(3, 0), (3, 1), (3, 2), (4, 3), (4, 4), (4, 5)]
AREAS_PER_LEVEL = 256
SCREEN = (32, 48)
VIDEO_FRAMES = 12
DARK_FRAMES = play.BREAK_FRAMES
SURVIVE = Part("Survive", "Survive", None)
LEVELS = [Part("Level 1", "Survive", "Boss"), Part("Level 2", "Boss", None)]


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


def route_taken() -> list[Segment]:
    return [Segment("Survive", [[1, 0]] * 4, b"", False)]


def taken_routes(tmp_path: Path, parts: list[Part]) -> Routes:
    routes = Routes(tmp_path / "levels.msgpack", tmp_path / "progress.msgpack", parts)
    routes.save(Leg("Survive", Start("Survive", bytes([0])), None), route_taken())
    return routes


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
    runner = SegmentRunner(
        LevelledWrapper(SavingEmulator(tmp_path / "recordings", LEVELLED_SCRIPT), fake_routes(tmp_path))
    )

    collected = samples.collect_samples(
        runner, [(runner.power_on(), route_taken())], [Start("Survive", bytes([3]))], [], 0
    )

    placed = [label for label in collected.labels if label[0] >= 0]
    assert sum(label[3] == 1 for label in placed) >= 20
    assert all(label[0] == label[3] * AREAS_PER_LEVEL for label in placed)
    assert collected.labels[-1] == (AREAS_PER_LEVEL, 5, 0, 1)


def test_places_the_search_explored_add_screens_of_their_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 1)
    runner = SegmentRunner(
        LevelledWrapper(SavingEmulator(tmp_path / "recordings", LEVELLED_SCRIPT), fake_routes(tmp_path))
    )

    collected = samples.collect_samples(runner, [], [], [Start("Survive", bytes([4]))], 0)

    assert (AREAS_PER_LEVEL, 5, 0, 1) in collected.labels


def stored_cell(key: list[object], state: int) -> StoredCell:
    return StoredCell(key, bytes([state]), None, 0, 0, 0, None, 0.0, 0)


def test_the_guide_starts_from_a_few_archived_cells_per_route_cell(tmp_path: Path):
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


def write_video(path: Path, levels: int) -> None:
    black = [np.zeros((32, 64, 3), np.uint8)] * DARK_FRAMES
    played = [np.full((32, 64, 3), 40 + index * 15, np.uint8) for index in range(VIDEO_FRAMES)]
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (64, 32))
    for screen in black + (played + black) * levels:
        video.write(screen)
    video.release()


def guided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, known: bool, levels: int):
    torch = pytest.importorskip("torch")
    from datenwissenschaften.reference import model

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 20)
    monkeypatch.setattr(model, "EPOCHS", 1)
    monkeypatch.setattr(model, "device", lambda: torch.device("cpu"))
    monkeypatch.setattr(model, "familiar", lambda _, reading: np.full(len(reading.places), known))
    config = write_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("guide: false", "guide: true"), encoding="utf-8")
    settings = load_config(config)
    reference_dir(settings).mkdir(parents=True)
    write_video(reference_dir(settings) / "video.avi", levels)
    return settings


def test_the_guide_reads_the_tas_video_into_a_route_that_counts_down_to_the_score_screen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = guided(tmp_path, monkeypatch, True, 1)

    guide.ensure_guide(settings, placed_runner(tmp_path), taken_routes(tmp_path, [SURVIVE]), None)

    trace = json.loads((reference_dir(settings) / TRACE_FILE).read_text(encoding="utf-8"))
    assert [step[0] for step in trace] == list(range(DARK_FRAMES, DARK_FRAMES + VIDEO_FRAMES))
    rows = json.loads(tas_route_path(settings).read_text(encoding="utf-8"))
    assert all(len(row) == 5 for row in rows) and min(row[4] for row in rows) >= 1
    assert not route_path(settings).exists()


def test_the_guide_is_built_once_and_never_over_an_existing_tas_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, True, 1)
    tas_route_path(settings).write_text("[]", encoding="utf-8")

    guide.ensure_guide(settings, cast(SegmentRunner, None), cast(Routes, None), None)

    assert tas_route_path(settings).read_text(encoding="utf-8") == "[]"


def test_the_guide_of_a_game_with_levels_waits_for_the_level_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, True, 1)

    guide.ensure_guide(settings, cast(SegmentRunner, None), taken_routes(tmp_path, [SURVIVE, *LEVELS]), None)

    assert not tas_route_path(settings).exists()


def test_a_tas_video_with_another_number_of_levels_than_the_game_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, True, 2)

    with pytest.raises(RuntimeError, match="2 levels"):
        guide.ensure_guide(settings, placed_runner(tmp_path), taken_routes(tmp_path, [SURVIVE]), None)


def test_a_tas_map_missing_a_level_is_never_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, False, 1)

    guide.ensure_guide(settings, placed_runner(tmp_path), taken_routes(tmp_path, [SURVIVE]), None)

    assert not tas_route_path(settings).exists()
    assert not (reference_dir(settings) / TRACE_FILE).exists()


def test_a_level_is_the_play_between_two_long_dark_screens_and_short_fades_stay_inside_it():
    dark = np.concatenate(
        [np.ones(200), np.zeros(50), np.ones(30), np.zeros(50), np.ones(200), np.zeros(40), np.ones(200)]
    )

    assert play.level_spans(dark.astype(bool), 2) == [(200, 330), (530, 570)]


def test_a_video_whose_dark_screens_frame_other_levels_than_the_game_has_fails():
    dark = np.concatenate([np.zeros(300), np.ones(200), np.zeros(300)]).astype(bool)

    with pytest.raises(RuntimeError, match="0 levels"):
        play.level_spans(dark, 1)


def test_a_score_screen_is_dark_but_a_dim_level_is_not():
    score = np.zeros((1, 32, 32, 3), np.uint8)
    score[0, :4] = 255
    dim = np.full((1, 32, 32, 3), (0, 0, 120), np.uint8)

    assert play.dark_screens(np.concatenate([score, dim])).tolist() == [True, False]
