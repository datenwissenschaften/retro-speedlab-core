from importlib.metadata import version
from pathlib import Path

import aim
from packaging.version import Version

FILELOCK_ADVISORIES_FIXED = Version("4.2.1")


def test_the_aim_repository_locks_with_a_filelock_that_has_its_advisories_fixed():
    assert Version(version("filelock")) >= FILELOCK_ADVISORIES_FIXED


def test_a_closed_aim_run_releases_its_lock_so_the_next_round_reopens_it(tmp_path: Path):
    run_hash = None
    rewards: list[list[float]] = []
    for step in range(2):
        run = aim.Run(
            run_hash=run_hash,
            repo=str(tmp_path),
            experiment="local",
            capture_terminal_logs=False,
            system_tracking_interval=None,
        )
        run.track(step, name="policy/reward", step=step)
        rewards = [list(metric.values.values_numpy()) for metric in run.metrics()]
        run_hash = run.hash
        run.close()

    assert rewards == [[0, 1]]
