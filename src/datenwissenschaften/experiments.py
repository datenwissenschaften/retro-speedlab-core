from typing import Any

import aim

from datenwissenschaften.settings import RetroSpeedlabConfig

JOB = "policy"


def start_experiment(settings: RetroSpeedlabConfig, hyperparameters: dict[str, Any]) -> aim.Run:
    release = settings.ui.release
    game = settings.training.game
    category = settings.speedrun.game.category
    run = aim.Run(repo=settings.experiments.repo, experiment=release, capture_terminal_logs=False)
    run.name = f"{JOB} {release}"
    run["job"] = JOB
    run["hparams"] = {
        "game": game,
        "category": category,
        "policy_emulators": settings.speedrun.policy_emulators,
        **hyperparameters,
    }
    for tag in (game, category):
        run.add_tag(tag)  # pyright: ignore[reportUnknownMemberType]
    return run
