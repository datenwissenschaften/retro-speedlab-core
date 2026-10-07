import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.wrapper import Observation, StateMachineGymWrapper
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig

STOP_SECONDS = 30.0


@dataclass(slots=True, frozen=True)
class PracticeStep:
    observation: Observation
    reward: float
    segment_ends: bool
    state: str


def practise(
    wrapper_cls: type[StateMachineGymWrapper],
    config: RetroSpeedlabConfig,
    speedrun: bool,
    worker: int,
    connection: Connection,
) -> None:
    setup_logging(config.log_level)
    env = make_environment(wrapper_cls, config, worker)
    env.speedrun = speedrun
    observation, info = env.reset()
    connection.send((observation, info["state"]))
    while (action := connection.recv()) is not None:
        state = info["state"]
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        segment_ends = done or info["state"] != state
        if done:
            observation, info = env.reset()
        connection.send(PracticeStep(observation, reward, segment_ends, info["state"]))
    env.close()


class PracticeEnvironments:
    def __init__(
        self, wrapper_cls: type[StateMachineGymWrapper], config: RetroSpeedlabConfig, speedrun: bool, workers: range
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.connections: list[Connection] = []
        self.processes: list[multiprocessing.process.BaseProcess] = []
        for worker in workers:
            parent, child = context.Pipe()
            process = context.Process(target=practise, args=(wrapper_cls, config, speedrun, worker, child), daemon=True)
            process.start()
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
        starts = [connection.recv() for connection in self.connections]
        self.observations: list[Observation] = [observation for observation, _ in starts]
        self.states: list[str] = [state for _, state in starts]

    def send(self, actions: list[int]) -> None:
        for connection, action in zip(self.connections, actions, strict=True):
            connection.send(action)

    def receive(self) -> list[PracticeStep]:
        steps: list[PracticeStep] = [connection.recv() for connection in self.connections]
        self.observations = [step.observation for step in steps]
        self.states = [step.state for step in steps]
        return steps

    def close(self) -> None:
        for connection in self.connections:
            connection.send(None)
        for process in self.processes:
            process.join(STOP_SECONDS)
            if process.is_alive():
                process.kill()
