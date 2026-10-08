import dataclasses
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.facts import Facts

PROBE_WORKER = 0
SEED_SUFFIX = ".state"


@dataclass(slots=True, frozen=True)
class Sample:
    state: str
    facts: Facts
    text: str
    question: str


def seeded_states(config: RetroSpeedlabConfig, state_names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in state_names if (config.paths.curriculum_dir / f"{name}{SEED_SUFFIX}").is_file())


def collect_samples(
    wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig, decisions: int, seed: int
) -> list[Sample]:
    with tempfile.TemporaryDirectory() as scratch:
        env = make_environment(wrapper_cls, isolated(config, Path(scratch)), PROBE_WORKER, False)
        try:
            states = seeded_states(config, env.curriculum.state_names)
            if not states:
                raise RuntimeError(f"No curriculum seeds in {config.paths.curriculum_dir} to probe from.")
            rng = np.random.default_rng(seed)
            return [sample for state in states for sample in play_from(env, state, decisions, rng)]
        finally:
            env.close()


def play_from(env: StateMachineGymWrapper, state: str, decisions: int, rng: np.random.Generator) -> list[Sample]:
    frame, _ = env.env.reset()
    env.start_from(state, frame)
    samples: list[Sample] = []
    for _ in range(decisions):
        ram = env.read_ram()
        observation = env.observer.observation(ram)
        samples.append(Sample(state, env.observer.facts(ram), observation["state"], observation["question"]))
        _, _, terminated, truncated, info = env.step(int(rng.integers(env.action_space.n)))
        if terminated or truncated or info["state"] != state:
            frame, _ = env.env.reset()
            env.start_from(state, frame)
    return samples


def isolated(config: RetroSpeedlabConfig, scratch: Path) -> RetroSpeedlabConfig:
    paths = dataclasses.replace(config.paths, cache_dir=scratch / "cache", record_dir=scratch / "recordings")
    return dataclasses.replace(config, paths=paths)
