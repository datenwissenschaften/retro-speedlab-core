from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import wandb
from speedlab_engine.cartridge import BUTTONS
from speedlab_engine.collection import DEMO_COLLECTION

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo, Shown, ram
from datenwissenschaften.route.route_map import RouteMap
from datenwissenschaften.states.state import State

FRAME_RATE = 60.0
FAKE_VERSION = "1.0.0"
NO_EXPERIMENT = wandb.init(project="tests", mode="disabled")
SAVING_SCRIPT = [(3, 0), (3, 1), (3, 2), (3, 5), (3, 6), (3, 6), (3, 6), (3, 6)]
FAKE_BUTTONS = ["A", "START"]


@dataclass
class FakeRam(RamInfo):
    shown = {"lives": Shown("Lives", False), "score": Shown("Score", False)}
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


def saved_state(position: int) -> bytes:
    return bytes([position])


class FakeEmulator:
    def __init__(self, script: list[tuple[int, int]]) -> None:
        self.gamename = "games.fake.cartridge"
        self.version = FAKE_VERSION
        self.buttons = FAKE_BUTTONS
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

    def reset(self) -> None:
        self.position = 0

    def step(self, action) -> None:
        self.pressed.append(np.asarray(action))
        self.position = min(self.position + 1, len(self.script) - 1)

    def get_ram(self) -> np.ndarray:
        return np.asarray(self.script[self.position], dtype=np.uint8)

    def action_to_array(self, action: np.ndarray) -> list[np.ndarray]:
        mask = np.zeros(len(BUTTONS), dtype=np.uint8)
        for name, pressed in zip(FAKE_BUTTONS, action, strict=True):
            mask[BUTTONS.index(name)] = pressed
        return [mask]


class SavingEmulator(FakeEmulator):
    def __init__(self, script: list[tuple[int, int]]) -> None:
        super().__init__(script)
        self.em.get_state = lambda: saved_state(self.position)
        self.em.set_state = self._load
        self.em.step = self._advance
        self.em.set_button_mask = lambda mask, player: None

    def _load(self, state: bytes) -> None:
        self.position = state[0]

    def _advance(self) -> None:
        self.position = min(self.position + 1, len(self.script) - 1)


class FakeWrapper(StateMachineGymWrapper[FakeRam]):
    start_state_cls = Survive
    state_classes = (Survive, Boss)
    ram_info_cls = FakeRam
    action_table = np.array([[[1, 0]], [[0, 1]]], dtype=np.int8)
    levels = ()


def fake_environment(tmp_path: Path, script: list[tuple[int, int]]) -> FakeWrapper:
    return FakeWrapper(FakeEmulator(script), fake_routes(tmp_path))


def saving_environment(tmp_path: Path) -> FakeWrapper:
    return FakeWrapper(SavingEmulator(SAVING_SCRIPT), fake_routes(tmp_path))


def fake_routes(tmp_path: Path) -> RouteMap:
    return RouteMap(tmp_path / "route.json")


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    (tmp_path / "hints.json").write_text("[]")
    config_path.write_text(
        "\n".join(
            [
                f"paths: {{collection: '{DEMO_COLLECTION}', models: models, cache: cache, lab_run: lab-run}}",
                "training: {game: walk}",
                "upload: {url: 'https://upload.test', api_key: null}",
                "ui: {enable: false, live: true, release: local}",
                "twitch: {enabled: true}",
                "agent: {stall_minutes: 30}",
                "speedrun:",
                "  scouts: 1",
                "  backtrackers: 0",
                "  beam_scouts: 0",
                "  vision_scouts: 0",
                "  polishers: 1",
                "  segment_seconds: 0.05",
                "  policy: false",
                "  policy_emulators: 1",
                "  category: Any%",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
