from pathlib import Path

from loguru import logger

from datenwissenschaften.accelerator import configure_accelerator
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.network import LayaNetwork
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.probe.readability import MIN_OFFSET_R2, Readability, readability
from datenwissenschaften.probe.samples import collect_samples
from datenwissenschaften.settings import load_config


class UnreadableFacts(RuntimeError):
    pass


def probe(wrapper_cls: type[StateMachineGymWrapper], config_path: Path, decisions: int, seed: int) -> None:
    config = load_config(config_path)
    setup_logging(config.log_level)
    samples = collect_samples(wrapper_cls, config, decisions, seed)
    network = LayaNetwork(config.laya.checkpoint, wrapper_cls.action_descriptions, configure_accelerator())
    results = readability(network, samples)
    for result in results:
        logger.info(f"{result.state:>12} {result.fact:<24} R² {result.r2:6.3f}{' required' if result.required else ''}")
    failures = [result for result in results if not result.passed]
    if failures:
        raise UnreadableFacts(f"Laya cannot read these offsets (R² < {MIN_OFFSET_R2}): {describe(failures)}")
    logger.info(f"Laya reads every offset of {len(samples)} probe decisions.")


def describe(results: list[Readability]) -> str:
    return ", ".join(f"{result.state}.{result.fact} {result.r2:.2f}" for result in results)
