from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeWrapper, write_config

from datenwissenschaften.evaluation import ab_test as ab_test_module
from datenwissenschaften.evaluation.ab_test import ABTest, evaluation_starts
from datenwissenschaften.evaluation.player import Attempt, Player
from datenwissenschaften.evaluation.snapshot import snapshot_models, with_models
from datenwissenschaften.evaluation.statistics import compare
from datenwissenschaften.settings import load_config
from datenwissenschaften.workflow import evaluation as evaluation_module

PHASES = ("Menu", "Survive", "Boss")


class ScriptedEnv:
    def __init__(self, steps: list[tuple[float, bool, str, bool]]) -> None:
        self.steps = steps
        self.starts: list[str | None] = []

    def reset_to(self, start: str | None) -> tuple[dict[str, str], dict]:
        self.starts.append(start)
        return {"state": "text"}, {"state": "Menu", "advice": None}

    def step(self, action: int) -> tuple[dict[str, str], float, bool, bool, dict]:
        reward, terminated, state, won = self.steps.pop(0)
        return {"state": "text"}, reward, terminated, False, {"state": state, "advice": 1, "won": won}


class GreedyModels:
    def __init__(self) -> None:
        self.explorations: list[dict[str, float]] = []

    def decide(self, observations: list, states: list[str], exploration: dict[str, float], advice: list) -> list:
        self.explorations.append(exploration)
        return [SimpleNamespace(action=0)]


def test_a_clear_improvement_has_a_positive_interval_and_a_high_probability():
    comparison = compare([1.0, 2.0, 1.5, 1.0], [5.0, 6.0, 5.5, 6.5])

    assert comparison.before == 1.375
    assert comparison.after == 5.75
    assert comparison.difference_low > 0
    assert comparison.probability_of_improvement == 1.0


def test_equal_models_are_a_coin_flip():
    comparison = compare([3.0, 3.0, 3.0], [3.0, 3.0, 3.0])

    assert comparison.difference_low == comparison.difference_high == 0.0
    assert comparison.probability_of_improvement == 0.5


def test_one_attempt_per_model_is_not_a_comparison():
    with pytest.raises(ValueError):
        compare([1.0], [2.0, 3.0])


def test_comparisons_name_their_metrics():
    metrics = compare([1.0, 2.0], [2.0, 3.0]).metrics("score")

    assert set(metrics) == {
        "score/before",
        "score/after",
        "score/difference_low",
        "score/difference_high",
        "score/probability_of_improvement",
    }


def test_the_player_plays_greedily_and_records_score_progress_and_win():
    env = ScriptedEnv([(1.0, False, "Survive", False), (2.0, True, "Boss", True)])
    models = GreedyModels()

    attempt = Player(env, models, PHASES).play("Survive", 10)

    assert attempt == Attempt(3.0, 2.0, True)
    assert env.starts == ["Survive"]
    assert models.explorations == [{"Menu": 0.0}, {"Survive": 0.0}]


def test_the_player_stops_after_its_decisions():
    env = ScriptedEnv([(1.0, False, "Survive", False)] * 5)

    attempt = Player(env, GreedyModels(), PHASES).play(None, 2)

    assert attempt == Attempt(2.0, 1.0, False)


def test_attempts_start_at_power_on_and_at_the_frontier_once_it_has_a_checkpoint():
    def env(frontier: str | None, checkpoint: bool) -> SimpleNamespace:
        curriculum = SimpleNamespace(active_state=lambda: frontier, has_checkpoint=lambda state: checkpoint)
        return SimpleNamespace(curriculum=SimpleNamespace(curriculum=curriculum))

    assert evaluation_starts(env(None, False)) == (None,)
    assert evaluation_starts(env("Boss", False)) == (None,)
    assert evaluation_starts(env("Boss", True)) == (None, "Boss")


def test_a_snapshot_copies_the_games_models_and_a_config_can_point_at_it(tmp_path: Path):
    config = load_config(write_config(tmp_path))
    model = config.paths.models_dir / "FakeGame-v0" / "Survive" / "laya.pt"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"weights")

    snapshot = snapshot_models(config)

    assert (snapshot / "FakeGame-v0" / "Survive" / "laya.pt").read_bytes() == b"weights"
    assert with_models(config, snapshot).paths.models_dir == snapshot
    assert with_models(config, snapshot).paths.cache_dir == config.paths.cache_dir


def test_an_ab_test_plays_both_models_from_the_same_starts_and_compares_them(tmp_path: Path, monkeypatch):
    config = load_config(write_config(tmp_path))
    played: list[tuple[Path, str | None]] = []
    closed: list[str] = []

    class FakeEnv:
        curriculum = SimpleNamespace(
            curriculum=SimpleNamespace(active_state=lambda: "Boss", has_checkpoint=lambda state: True)
        )
        observer = SimpleNamespace(advisor=None)

        def close(self) -> None:
            closed.append("env")

    class FakeModels:
        def __init__(self, models_dir: Path) -> None:
            self.models_dir = models_dir

        def close(self) -> None:
            closed.append("models")

    class FakePlayer:
        def __init__(self, env: FakeEnv, models: FakeModels, phases: tuple[str, ...]) -> None:
            self.models = models

        def play(self, start: str | None, decisions: int) -> Attempt:
            played.append((self.models.models_dir, start))
            better = self.models.models_dir == config.paths.models_dir
            return Attempt(10.0 if better else 1.0, 2.0 if better else 1.0, better)

    class FakeAdvisors:
        def __init__(self, actions: tuple[str, ...], device: str, path: object) -> None:
            pass

        def advise(self, state: str, inputs: object) -> None:
            pass

        def close(self) -> None:
            closed.append("advisors")

    monkeypatch.setattr(ab_test_module, "make_environment", lambda wrapper, config, worker, main: FakeEnv())
    monkeypatch.setattr(ab_test_module, "laya_network", lambda wrapper, config: "network")
    monkeypatch.setattr(
        ab_test_module, "state_models", lambda wrapper, network, context: FakeModels(context.config.paths.models_dir)
    )
    monkeypatch.setattr(ab_test_module, "configure_accelerator", lambda: "cpu")
    monkeypatch.setattr(ab_test_module, "Advisors", FakeAdvisors)
    monkeypatch.setattr(ab_test_module, "Player", FakePlayer)

    comparisons = ABTest(FakeWrapper, config).run(tmp_path / "before")

    episodes = ab_test_module.EPISODES
    assert played == [
        *[(tmp_path / "before", None)] * episodes,
        *[(tmp_path / "before", "Boss")] * episodes,
        *[(config.paths.models_dir, None)] * episodes,
        *[(config.paths.models_dir, "Boss")] * episodes,
    ]
    assert comparisons["score"].probability_of_improvement == 1.0
    assert comparisons["progress"].after == 2.0
    assert closed == ["advisors", "models", "advisors", "models", "env"]


def test_the_ab_test_task_removes_the_snapshot_even_when_the_test_fails(tmp_path: Path, monkeypatch):
    before = tmp_path / "before"
    before.mkdir()

    class FailingTest:
        def __init__(self, wrapper_cls: type, config: object) -> None:
            pass

        def run(self, before_dir: Path) -> None:
            raise RuntimeError("emulator crashed")

    monkeypatch.setattr(evaluation_module, "ABTest", FailingTest)

    with pytest.raises(RuntimeError):
        evaluation_module.ab_test("fakes:FakeWrapper", str(write_config(tmp_path)), str(before))

    assert not before.exists()
