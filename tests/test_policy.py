import random
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
from fakes import NO_EXPERIMENT
from stable_baselines3.common.vec_env import DummyVecEnv
from test_tas import START, LineRunner

from datenwissenschaften.policy import env as env_module
from datenwissenschaften.policy import starts as starts_module
from datenwissenschaften.policy import trainer as trainer_module
from datenwissenschaften.policy.env import EPISODE_DECISIONS, EXIT, PolicyEnv
from datenwissenschaften.policy.prior import POLICY_FILE, PolicyPrior, observation
from datenwissenschaften.policy.starts import openings
from datenwissenschaften.policy.trainer import RoundCallback, export_policy, policy_model
from datenwissenschaften.speedrun.archive_store import save_archive
from datenwissenschaften.speedrun.archives import archive_key
from datenwissenschaften.speedrun.live import read_reports
from datenwissenschaften.speedrun.novelty import RamNovelty
from datenwissenschaften.speedrun.policy_start import (
    POLICY_EXITS_FILE,
    POLICY_START_FILE,
    load_policy_exits,
    save_policy_start,
)
from datenwissenschaften.speedrun.preferences import Preferences
from datenwissenschaften.speedrun.prior import ActionPrior
from datenwissenschaften.speedrun.runner import Start
from datenwissenschaften.speedrun.search import UNBOUNDED, Cell, Trail

CPU = torch.device("cpu")
RIGHT, LEFT = 1, 2
SCREEN = (224, 240, 3)


class PoweredLine(LineRunner):
    def __init__(self) -> None:
        super().__init__()
        self.emulator = SimpleNamespace(em=SimpleNamespace(get_screen=lambda: np.zeros(SCREEN, dtype=np.uint8)))

    def power_on(self) -> Start:
        self.restore(START)
        return START


@pytest.fixture
def line_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PolicyEnv:
    monkeypatch.setattr(env_module, "worker_runner", lambda wrapper, config: PoweredLine())
    save_policy_start(tmp_path / POLICY_START_FILE, START)
    (tmp_path / "live").mkdir()
    return PolicyEnv("fakes:FakeWrapper", tmp_path / "config.yaml", tmp_path, tmp_path / "live")


def test_an_episode_starts_where_the_search_stands_and_ends_on_the_way_out(line_env: PolicyEnv):
    line_env.reset(seed=1)
    steps = [line_env.step(np.int64(RIGHT)) for _ in range(3)]

    assert [step[2] for step in steps] == [False, False, True]
    state, pressed = steps[-1][4][EXIT]
    assert state == "Walk" and len(pressed) == 6
    assert steps[-1][1] > steps[0][1]


def test_an_episode_ends_by_death_or_after_its_decisions(line_env: PolicyEnv):
    line_env.reset(seed=1)
    _, reward, terminated, _, info = line_env.step(np.int64(LEFT))
    _, reward, terminated, _, info = line_env.step(np.int64(LEFT))
    assert terminated and reward < 0 and EXIT not in info

    line_env.reset(seed=1)
    truncated = [line_env.step(np.int64(0))[3] for _ in range(EPISODE_DECISIONS)]
    assert truncated[-1] and not any(truncated[:-1])


def test_the_search_reads_the_policy_the_trainer_exported(line_env: PolicyEnv, tmp_path: Path):
    model = policy_model(DummyVecEnv([lambda: line_env]), CPU)
    memory = np.asarray([3], dtype=np.uint8)
    export_policy(model.policy, tmp_path / POLICY_FILE)

    probabilities = PolicyPrior(tmp_path / POLICY_FILE).probabilities(memory, 3)

    with torch.no_grad():
        expected = model.policy.get_distribution(torch.from_numpy(observation(memory))[None]).distribution.probs
    assert np.allclose(probabilities, expected[0].numpy(), atol=1e-5)
    assert PolicyPrior(tmp_path / "none.npz").choose(memory, 3, random.Random(1)) is None


def test_ppo_rounds_export_the_policy_and_keep_the_shortest_way_out(
    line_env: PolicyEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(trainer_module, "ROLLOUT_DECISIONS", 64)
    monkeypatch.setattr(trainer_module, "MINIBATCH", 32)
    model = policy_model(DummyVecEnv([lambda: line_env]), CPU)
    callback = RoundCallback(tmp_path, NO_EXPERIMENT)

    model.learn(128, callback=callback)

    assert callback.rounds == 1
    assert PolicyPrior(tmp_path / POLICY_FILE).probabilities(np.zeros(1, dtype=np.uint8), 3) is not None
    assert [report.mode for report in read_reports(tmp_path / "live")] == ["policy"]
    exits = load_policy_exits(tmp_path / POLICY_EXITS_FILE)
    assert "Walk" not in exits or len(exits["Walk"]) >= 6


def test_the_search_follows_the_table_or_the_policy_part_of_the_time(line_env: PolicyEnv, tmp_path: Path):
    (tmp_path / "prior.msgpack").write_bytes(msgspec.msgpack.encode({"Walk": {2: 1}}))
    export_policy(policy_model(DummyVecEnv([lambda: line_env]), CPU).policy, tmp_path / POLICY_FILE)
    preferences = Preferences(ActionPrior(tmp_path / "prior.msgpack"), PolicyPrior(tmp_path / POLICY_FILE))
    rng = random.Random(3)

    chosen = [preferences.choose(LineRunner(), 3, rng) for _ in range(200)]

    assert chosen.count(None) > 50 and chosen.count(2) > 50
    assert {choice for choice in chosen if choice is not None} <= {0, 1, 2}


def place(x: int, score: float) -> Cell:
    trail = Trail(None, np.full((x, 1), RIGHT, dtype=np.int8))
    return Cell(Start("Walk", x.to_bytes(2, "big", signed=True), {}), trail, x, 0, None, score, 0)


def save_places(tmp_path: Path, cells: list[Cell]) -> None:
    folder = tmp_path / "archives" / archive_key(START, UNBOUNDED)
    folder.mkdir(parents=True)
    keyed = {("Walk", cell.frames): cell for cell in cells}
    save_archive(folder / "1.msgpack", "Walk", keyed, RamNovelty(np.zeros(1, dtype=np.uint8)))


def test_the_openings_are_the_search_start_then_its_places_closest_to_the_exit_first(tmp_path: Path):
    save_places(tmp_path, [place(2, 1.0), place(4, 3.0)])

    found = openings(tmp_path / "archives", START, 1)

    assert [opening.start for opening in found] == [START, place(4, 3.0).start, place(2, 1.0).start]
    assert [len(opening.prefix) for opening in found] == [0, 4, 2]


def test_an_episode_from_a_place_inside_the_part_hands_the_search_a_way_out_from_its_start(
    line_env: PolicyEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    save_places(tmp_path, [place(4, 3.0)])
    monkeypatch.setattr(starts_module, "FRONTIER_SHARE", 1.0)

    line_env.reset(seed=1)
    _, _, terminated, _, info = line_env.step(np.int64(RIGHT))

    state, pressed = info[EXIT]
    assert terminated and state == "Walk"
    assert pressed.tolist() == [[RIGHT]] * 6
