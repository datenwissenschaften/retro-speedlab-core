import shutil
from pathlib import Path

from datenwissenschaften.evaluation.ab_test import ABTest
from datenwissenschaften.evaluation.snapshot import snapshot_models
from datenwissenschaften.settings import load_config
from datenwissenschaften.workflow.environment import speedlab
from datenwissenschaften.workflow.wrapper import wrapper_class


@speedlab.task
def snapshot(config: str) -> str:
    return str(snapshot_models(load_config(Path(config))))


@speedlab.task
def ab_test(wrapper: str, config: str, before: str) -> None:
    try:
        ABTest(wrapper_class(wrapper), load_config(Path(config))).run(Path(before))
    finally:
        shutil.rmtree(before)
