import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import FakeRam, FakeWrapper, SavingEmulator, fake_routes, write_config

from datenwissenschaften.reference import guide, samples
from datenwissenschaften.reference.paths import TRACE_FILE, reference_dir, route_path
from datenwissenschaften.route.position import Position
from datenwissenschaften.settings import load_config
from datenwissenschaften.tas.movie import Segment
from datenwissenschaften.tas.runner import SegmentRunner

SCRIPT = [(3, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
SCREEN = (32, 48)
VIDEO_FRAMES = 12


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


def route_taken() -> list[Segment]:
    return [Segment("Survive", [[1, 0]] * 4, b"", False)]


def test_samples_label_every_screen_with_its_place_and_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 20)

    collected = samples.collect_samples(placed_runner(tmp_path), route_taken(), 0)

    assert len(collected.labels) >= 20 and len(collected.screens) == len(collected.labels)
    assert collected.screens[0].shape == (SCREEN[0] // 2, SCREEN[1] // 2, 3)
    assert {label[3] for label in collected.labels} == {0}
    assert collected.labels[0] == (0, 32, 48, 0)


def guided(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, playing: float):
    torch = pytest.importorskip("torch")
    from datenwissenschaften.reference import model

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(samples, "SAMPLES_PER_LEVEL", 20)
    monkeypatch.setattr(model, "EPOCHS", 1)
    monkeypatch.setattr(model, "PLAYING", playing)
    monkeypatch.setattr(model, "device", lambda: torch.device("cpu"))
    config = write_config(tmp_path)
    config.write_text(config.read_text(encoding="utf-8").replace("guide: false", "guide: true"), encoding="utf-8")
    settings = load_config(config)
    reference_dir(settings).mkdir(parents=True)
    video = cv2.VideoWriter(str(reference_dir(settings) / "video.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 60, (64, 32))
    for index in range(VIDEO_FRAMES):
        video.write(np.full((32, 64, 3), index * 20, np.uint8))
    video.release()
    return settings


def test_the_guide_reads_the_tas_video_into_a_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, 0.0)

    guide.ensure_guide(settings, placed_runner(tmp_path), route_taken())

    trace = json.loads((reference_dir(settings) / TRACE_FILE).read_text(encoding="utf-8"))
    assert [step[0] for step in trace] == list(range(VIDEO_FRAMES)) and all(len(step) == 4 for step in trace)
    assert all(len(row) == 5 for row in json.loads(route_path(settings).read_text(encoding="utf-8")))


def test_a_tas_video_never_seen_in_play_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = guided(tmp_path, monkeypatch, 1.0)

    with pytest.raises(RuntimeError, match="in play"):
        guide.ensure_guide(settings, placed_runner(tmp_path), route_taken())
