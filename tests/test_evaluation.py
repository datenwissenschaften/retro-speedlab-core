from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest
from fakes import write_config

from datenwissenschaften.evaluation import branch_play
from datenwissenschaften.evaluation import duel as duel_module
from datenwissenschaften.evaluation.duel import DRAW, Entrant, decide, probability_better
from datenwissenschaften.evaluation.player import Attempt, Player
from datenwissenschaften.settings import load_config

PHASES = ("Menu", "Survive", "Boss")


class ScriptedEnv:
    def __init__(self, steps: list[tuple[bool, str, bool]]) -> None:
        self.steps = steps
        self.starts: list[str | None] = []

    def reset_to(self, start: str | None) -> tuple[dict[str, str], dict]:
        self.starts.append(start)
        return {"state": "text"}, {"state": "Menu", "advice": None}

    def step(self, action: int) -> tuple[dict[str, str], float, bool, bool, dict]:
        terminated, state, won = self.steps.pop(0)
        return {"state": "text"}, 1.0, terminated, False, {"state": state, "advice": 1, "won": won}


class GreedyModels:
    def __init__(self) -> None:
        self.explorations: list[dict[str, float]] = []

    def decide(self, observations: list, states: list[str], exploration: dict[str, float], advice: list) -> list:
        self.explorations.append(exploration)
        return [SimpleNamespace(action=0)]


def attempts(*progress: float) -> list[Attempt]:
    return [Attempt(value, value == 1.0, 100) for value in progress]


def test_the_player_plays_greedily_from_power_on_and_records_progress_win_and_decisions():
    env = ScriptedEnv([(False, "Survive", False), (True, "Boss", True)])
    models = GreedyModels()

    attempt = Player(env, models, PHASES).play(10)

    assert attempt == Attempt(1.0, True, 2)
    assert env.starts == [None]
    assert models.explorations == [{"Menu": 0.0}, {"Survive": 0.0}]


def test_the_player_stops_after_its_decisions():
    env = ScriptedEnv([(False, "Survive", False)] * 5)

    assert Player(env, GreedyModels(), PHASES).play(2) == Attempt(0.5, False, 2)


def test_clearly_better_attempts_are_better_for_certain_and_equal_ones_are_a_coin_flip():
    assert probability_better([0.1, 0.2, 0.1], [0.8, 0.9, 0.7]) == 1.0
    assert probability_better([0.5, 0.5], [0.5, 0.5]) == 0.5


def test_wins_decide_the_duel_before_progress():
    verdict = decide([Entrant("claude", attempts(0.9, 0.9)), Entrant("codex", attempts(1.0, 0.1))])

    assert verdict.winner == "codex"


def test_without_wins_the_furthest_progress_decides():
    verdict = decide([Entrant("claude", attempts(0.6, 0.5)), Entrant("codex", attempts(0.2, 0.1))])

    assert verdict.winner == "claude"
    assert verdict.confidence == 1.0


def test_equal_wins_are_decided_by_fewer_decisions():
    faster = [Attempt(1.0, True, 50), Attempt(1.0, True, 60)]
    verdict = decide([Entrant("claude", attempts(1.0, 1.0)), Entrant("codex", faster)])

    assert verdict.winner == "codex"


def test_identical_play_is_a_draw():
    verdict = decide([Entrant("claude", attempts(0.5, 0.5)), Entrant("codex", attempts(0.5, 0.5))])

    assert verdict.winner == DRAW
    assert verdict.confidence == 0.5


def test_a_crashed_branch_forfeits_and_without_any_branch_there_is_no_winner():
    assert decide([Entrant("codex", attempts(0.1, 0.1))]).winner == "codex"
    with pytest.raises(RuntimeError):
        decide([])


def test_a_branch_plays_with_its_own_integrations_and_the_trained_models(tmp_path: Path):
    config = write_config(tmp_path)

    settings = branch_play.branch_config(config, tmp_path / "game-codex")

    assert settings.paths.integrations_dir == tmp_path / "game-codex" / "integrations"
    assert settings.paths.models_dir == load_config(config).paths.models_dir


def test_the_judge_plays_both_branches_and_a_crashed_one_loses(tmp_path: Path, monkeypatch, capsys):
    config = write_config(tmp_path)
    project = tmp_path / "game"
    played: list[Path] = []
    recorded: list[str] = []

    def run(command: list[str], cwd: Path, stdout: object, check: bool) -> None:
        played.append(cwd)
        if cwd.name.endswith("claude"):
            raise duel_module.subprocess.CalledProcessError(1, command)
        progress = 0.8
        Path(command[-1]).parent.mkdir(parents=True, exist_ok=True)
        Path(command[-1]).write_bytes(msgspec.json.encode(attempts(progress, progress)))

    monkeypatch.setattr(duel_module.subprocess, "run", run)
    monkeypatch.setattr(duel_module, "record", lambda settings, entrants, verdict: recorded.append(verdict.winner))

    duel_module.judge("game:Wrapper", config, project)

    assert played == [tmp_path / "game-claude", tmp_path / "game-codex"]
    assert recorded == ["codex"]
    assert capsys.readouterr().out == "codex\n"


def test_a_duel_without_tracking_fails(tmp_path: Path):
    settings = load_config(write_config(tmp_path))
    entrants = [Entrant("claude", attempts(0.5, 0.5)), Entrant("codex", attempts(0.5, 0.5))]

    with pytest.raises(RuntimeError):
        duel_module.record(settings, entrants, decide(entrants))
