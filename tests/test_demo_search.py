from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import msgspec
import numpy as np
from fakes import write_config
from speedlab_engine.cartridge import BUTTONS
from speedlab_engine.movie import Movie, decode

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo, Shown, ram
from datenwissenschaften.route.position import Position
from datenwissenschaften.speedrun import run, workers
from datenwissenschaften.speedrun.routes import FULL_RUN, Leg
from datenwissenschaften.states.state import State

HOLD_FRAMES = 8
GOAL_X = 200
DOT_Y = 120
WRAPPER = "test_demo_search:WalkWrapper"


def held(name: str | None) -> np.ndarray:
    return np.array([[name is not None and button == name for button in BUTTONS]] * HOLD_FRAMES, dtype=np.int8)


@dataclass
class WalkRam(RamInfo):
    shown = {"x": Shown("X", False)}
    x: int = ram(0)
    won: int = ram(1)

    def describe(self) -> dict[str, int]:
        return {"x": self.x}

    def position(self) -> Position:
        return Position(0, self.x, DOT_Y, 0, False)

    def progress(self) -> int:
        return self.x


class Walk(State[WalkRam]):
    goal = "Reach the goal"

    def _on_reset(self) -> None:
        self.walked = self.ram.x

    def _reward(self) -> float:
        gained, self.walked = self.ram.x - self.walked, self.ram.x
        return float(max(0, gained))

    def _won(self) -> bool:
        return self.ram.won == 1


class WalkWrapper(StateMachineGymWrapper[WalkRam]):
    start_state_cls = Walk
    state_classes = (Walk,)
    ram_info_cls = WalkRam
    action_table = np.stack([held("RIGHT"), held("LEFT"), held(None)])
    levels = ()


class SyncPool:
    def submit(self, function, *arguments) -> Future:
        future: Future = Future()
        future.set_result(function(*arguments))
        return future


class IdlePool:
    def submit(self, function, *arguments) -> Future:
        return Future()


def test_the_tas_search_beats_the_demo_cartridge_and_its_movie_renders(tmp_path: Path):
    tas = run.tas_run(WRAPPER, write_config(tmp_path), SyncPool(), IdlePool())
    power_on = tas.runner.power_on()

    segments = tas.extend(Leg(FULL_RUN, power_on, None), [], None, run.never_rest)
    run.publish(tas, segments)

    assert segments[-1].won and segments[-1].emulator_state[0] == GOAL_X
    assert run.frames(segments) >= GOAL_X - power_on.emulator_state[0]
    shown = msgspec.msgpack.decode((tas.output / "screen.movie").read_bytes(), type=Movie)
    assert (shown.cartridge, shown.version) == ("games.walk.cartridge", "1.0.0")
    assert len(decode(shown.frames)) == run.frames(segments)
    assert (tas.output / "screen.mp4").stat().st_size > 0
    assert tas.feed.learned.sprites[0]["name"] == "dot"


def test_the_beam_scout_walks_the_demo_dot_to_its_goal(tmp_path: Path):
    config = write_config(tmp_path)
    start = workers.worker_runner(WRAPPER, config).power_on()

    searched = workers.beam(WRAPPER, config, start, 1, 30.0, tmp_path / "live")

    assert searched.crossings and searched.crossings[0].won
    assert (tmp_path / "live" / "1.jpg").is_file()


def test_the_vision_scout_is_told_the_goal_and_what_the_lab_knows(tmp_path: Path, monkeypatch):
    config = write_config(tmp_path)
    mechanics = "[{title: Walk, text: Right walks.}]"
    (tmp_path / "knowledge.yaml").write_text(
        f"{{mechanics: {mechanics}, player: null, enemies: [], powerups: [], levels: [], ram: []}}"
    )
    asked: list[tuple[str, str]] = []
    monkeypatch.setattr(workers, "vision_search", lambda runner, start, goal, notes, live: asked.append((goal, notes)))
    start = workers.worker_runner(WRAPPER, config).power_on()

    workers.vision(WRAPPER, config, start, 1, None)

    assert asked == [("Reach the goal", "- Walk: Right walks.")]
