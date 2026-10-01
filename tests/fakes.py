from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch
from torch import nn

from datenwissenschaften.environment.curriculum_run import CurriculumRun
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.ram import RamInfo, ram
from datenwissenschaften.states.landmarks import Landmarks
from datenwissenschaften.states.state import State

VOCABULARY = 64
HIDDEN = 8
ACTIONS = {"left": "move left", "right": "move right"}

PRETRAINED_SEED = 7


class FakeTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3
    pad_token_id = 0

    def __call__(self, text: str, **kwargs) -> dict[str, list[int]]:
        ids = [4 + ord(character) % (VOCABULARY - 4) for character in text]
        return {"input_ids": ids[: kwargs["max_length"]] if "max_length" in kwargs else ids}


class FakeEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCABULARY, HIDDEN)
        self.checkpointing = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids)


class FakeDecision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = FakeEncoder()
        self.scorer = nn.Linear(HIDDEN, 1)
        self.head_checkpointing = False

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        hidden = self.encoder(input_ids)
        markers = torch.gather(hidden, 1, marker_pos[:, :, None].expand(-1, -1, HIDDEN))
        logits = self.scorer(markers).squeeze(-1).masked_fill(~marker_mask, -1e4)
        return logits, torch.zeros(len(input_ids), 2)


def fake_laya_load(checkpoint: str, device: str) -> SimpleNamespace:
    torch.manual_seed(PRETRAINED_SEED)
    return SimpleNamespace(tok=FakeTokenizer(), cfg={"max_len": 128, "head_max_len": 48}, model=FakeDecision())


@dataclass
class FakeRam(RamInfo):
    lives: int = ram(0)
    score: int = ram(1)


class Survive(State[FakeRam]):
    description = "Which move survives?"

    def _reward(self) -> float:
        return float(self.ram.score)

    def _terminated(self) -> bool:
        return self.ram.lives == 0

    def _won(self) -> bool:
        return self.ram.score >= 9

    def _next(self):
        return Boss if self.ram.score == 5 else None


class Boss(State[FakeRam]):
    description = "Which move beats the boss?"

    def _won(self) -> bool:
        return self.ram.score >= 7


class FakeEmulator(gym.Env):
    def __init__(self, movie_path: Path, script: list[tuple[int, int]]) -> None:
        self.movie_path = str(movie_path)
        self.movie_id = 0
        self.gamename = "FakeGame-v0"
        self.statename = "Level1.state"
        self.movie = SimpleNamespace(set_state=lambda state: None)
        self.em = SimpleNamespace(
            get_state=lambda: b"emulator", set_state=lambda state: None, get_screen_rate=lambda: 60.0
        )
        self.data = SimpleNamespace(reset=lambda: None, update_ram=lambda: None)
        self.action_space = gym.spaces.MultiBinary(2)
        self.observation_space = gym.spaces.Box(0, 255, (2, 2, 3), np.uint8)
        self.script = script
        self.position = 0
        self.pressed: list[np.ndarray] = []

    def reset(self, **kwargs):
        self.movie_id += 1
        self.position = 0
        return np.zeros((2, 2, 3), np.uint8), {}

    def step(self, action):
        self.pressed.append(np.asarray(action))
        self.position = min(self.position + 1, len(self.script) - 1)
        return np.zeros((2, 2, 3), np.uint8), 0.0, False, False, {}

    def get_ram(self) -> np.ndarray:
        return np.asarray(self.script[self.position], dtype=np.uint8)

    def render(self) -> np.ndarray:
        return np.zeros((4, 4, 3), np.uint8)


class FakeWrapper(StateMachineGymWrapper[FakeRam]):
    start_state_cls = Survive
    state_classes = (Survive, Boss)
    ram_info_cls = FakeRam
    action_table = np.array([[[1, 0]], [[0, 1]]], dtype=np.int8)
    action_descriptions = ACTIONS


def fake_environment(tmp_path: Path, script: list[tuple[int, int]]) -> FakeWrapper:
    emulator = FakeEmulator(tmp_path / "recordings", script)
    curriculum = CurriculumRun(tmp_path / "curriculum", ("Survive", "Boss"), "Level1")
    return FakeWrapper(emulator, curriculum, Landmarks(tmp_path / "landmarks.json"), "Level1")


def write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths: {roms: roms, savestates: savestates, models: models, recordings: recordings, cache: cache,"
                " database: database.json, reports: reports}",
                "training: {game: FakeGame-v0, savestates: [Level1], rotation_minutes: 60, fingerprint: null}",
                "laya: {checkpoint: fake/laya}",
                "upload: {url: 'https://upload.test', api_key: null}",
                "ui: {enable: false, host: 127.0.0.1, port: 18080, max_episodes: 10, release: local, persona: Retra}",
                "twitch: {enabled: true, dialogs: false, dialog_model: test/model:free}",
                "log_level: INFO",
            ]
        ),
        encoding="utf-8",
    )
    return config_path
