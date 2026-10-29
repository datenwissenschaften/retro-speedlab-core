from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften.environment.level_drill import LevelDrill
from datenwissenschaften.laya.imitation import Example
from datenwissenschaften.ppo import ENTROPY_FLOOR, ENTROPY_WEIGHT, MAX_ENTROPY_BOOST, entropy_weight
from datenwissenschaften.training.collapse_hook import COLLAPSED_UPDATES, CollapseHook
from datenwissenschaften.training.own_wins import OwnWins

ORDER = ("Menu", "Plateau", "Feed")
VIEW = {"state": "{}", "question": "Which move?"}


def test_a_collapsing_policy_gets_a_larger_entropy_bonus_up_to_a_limit():
    assert entropy_weight(0.5) == ENTROPY_WEIGHT
    assert entropy_weight(ENTROPY_FLOOR / 2) == 2 * ENTROPY_WEIGHT
    assert entropy_weight(0.0) == MAX_ENTROPY_BOOST * ENTROPY_WEIGHT


def test_only_attempts_that_leave_a_learning_state_forward_are_kept_and_saved(tmp_path: Path):
    path = lambda state: tmp_path / state / "wins.json"  # noqa: E731
    wins = OwnWins(ORDER, path, lambda state: state == "Plateau")

    wins.add(0, "Plateau", VIEW, 1, "Plateau", False)
    wins.add(0, "Plateau", VIEW, 2, "Menu", False)
    wins.add(1, "Plateau", VIEW, 3, "Plateau", True)
    wins.add(2, "Plateau", VIEW, 4, "Plateau", False)
    wins.add(2, "Plateau", VIEW, 5, "Feed", False)
    wins.add(0, "Feed", VIEW, 6, "Menu", False)

    assert [example.action for example in wins.examples("Plateau")] == [4, 5]
    assert OwnWins(ORDER, path, lambda state: True).examples("Plateau") == [
        Example("{}", "Which move?", 4),
        Example("{}", "Which move?", 5),
    ]
    assert wins.examples("Feed") == []


def test_a_drilled_state_whose_policy_stays_collapsed_starts_again_once(tmp_path: Path):
    resets: list[str] = []
    models = SimpleNamespace(
        agent=SimpleNamespace(entropy_share=0.0),
        require_active=lambda: "Plateau",
        reset=resets.append,
    )
    drill = LevelDrill(tmp_path)
    drill.state = "Plateau"
    hook = CollapseHook(models, drill)

    for _ in range(2 * COLLAPSED_UPDATES):
        hook.on_update()

    assert resets == ["Plateau"]
    models.agent.entropy_share = 1.0
    hook.on_update()
    assert hook.collapsed["Plateau"] == 0
