from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo, ram
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.states.state import State

FRAME_RATE = 60.0


@dataclass
class FakeRam(RamInfo):
    lives: int = ram(0)
    score: int = ram(1)

    def describe(self) -> dict[str, int]:
        return {"lives": self.lives, "score": self.score}


class Survive(State[FakeRam]):
    goal = "Survive"

    def _reward(self) -> float:
        return float(self.ram.score)

    def _terminated(self) -> bool:
        return self.ram.lives == 0

    def _won(self) -> bool:
        return self.ram.score >= 9

    def _next(self):
        return Boss if self.ram.score == 5 else None


class Boss(State[FakeRam]):
    goal = "Beat the boss"

    def _won(self) -> bool:
        return self.ram.score >= 7


class FakeEmulator:
    def __init__(self, movie_path: Path, script: list[tuple[int, int]]) -> None:
        self.movie_path = str(movie_path)
        self.movie_id = 0
        self.gamename = "FakeGame-v0"
        self.statename = "Level1.state"
        self.buttons = ["A", "START"]
        self.movie = SimpleNamespace(set_state=lambda state: None)
        self.em = SimpleNamespace(
            get_state=lambda: b"emulator",
            set_state=lambda state: None,
            get_screen_rate=lambda: FRAME_RATE,
            get_screen=lambda: np.zeros((4, 4, 3), np.uint8),
        )
        self.data = SimpleNamespace(reset=lambda: None, update_ram=lambda: None)
        self.script = script
        self.position = 0
        self.pressed: list[np.ndarray] = []

    def reset(self) -> tuple[np.ndarray, dict]:
        self.movie_id += 1
        self.position = 0
        return np.zeros((2, 2, 3), np.uint8), {}

    def step(self, action):
        self.pressed.append(np.asarray(action))
        self.position = min(self.position + 1, len(self.script) - 1)
        return np.zeros((2, 2, 3), np.uint8), 0.0, False, False, {}

    def get_ram(self) -> np.ndarray:
        return np.asarray(self.script[self.position], dtype=np.uint8)

    def stop_record(self) -> None:
        self.movie_path = None

    def auto_record(self, path: str) -> None:
        self.movie_path = path

    def render(self) -> np.ndarray:
        return np.zeros((4, 4, 3), np.uint8)


class FakeWrapper(StateMachineGymWrapper[FakeRam]):
    start_state_cls = Survive
    state_classes = (Survive, Boss)
    ram_info_cls = FakeRam
    action_table = np.array([[[1, 0]], [[0, 1]]], dtype=np.int8)
    levels = ()


def fake_environment(tmp_path: Path, script: list[tuple[int, int]]) -> FakeWrapper:
    emulator = FakeEmulator(tmp_path / "recordings", script)
    return FakeWrapper(emulator, fake_routes(tmp_path))


def fake_routes(tmp_path: Path) -> RouteMap:
    return RouteMap(lambda state: tmp_path / "models" / state / "route.json")


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths: {roms: roms, integrations: integrations, models: models, cache: cache, lab_run: lab-run}",
                "training: {game: FakeGame-v0}",
                "upload: {url: 'https://upload.test', api_key: null}",
                "ui: {enable: false, release: local}",
                "twitch: {enabled: true}",
                "agent: {stall_minutes: 30}",
                "speedrun:",
                "  workers: 1",
                "  segment_seconds: 0.05",
                "  id: fakegame-v0",
                "  title: Fake Game",
                "  platform: NES",
                "  category: Any%",
                "  speedrun_com: {game: fake, category: any, variables: {level: one}}",
                "  tasvideos: 7",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
