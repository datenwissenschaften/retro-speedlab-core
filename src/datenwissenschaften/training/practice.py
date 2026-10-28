import multiprocessing
from dataclasses import dataclass
from multiprocessing.connection import Connection

import numpy as np

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.logger import setup_logging
from datenwissenschaften.settings import RetroSpeedlabConfig
from datenwissenschaften.states.facts import Facts

STOP_SECONDS = 30.0


@dataclass(slots=True, frozen=True)
class PracticeView:
    inputs: np.ndarray
    facts: Facts
    question: str


@dataclass(slots=True, frozen=True)
class PracticeStep:
    view: PracticeView
    reward: float
    terminal: bool
    truncated: bool
    reached: str
    ended: bool
    state: str


@dataclass(slots=True, frozen=True)
class Restart:
    state: str
    emulator_state: bytes


def practice_view(env: StateMachineGymWrapper) -> PracticeView:
    facts = env.observer.facts(env.read_ram())
    return PracticeView(env.observer.encoded(facts), facts, env.state_machine.question)


def practise(
    wrapper_cls: type[StateMachineGymWrapper],
    config: RetroSpeedlabConfig,
    speedrun: bool,
    worker: int,
    connection: Connection,
) -> None:
    setup_logging(config.log_level)
    env = make_environment(wrapper_cls, config, worker, False)
    env.speedrun = speedrun
    _, info = env.reset()
    connection.send((practice_view(env), info["state"]))
    while (message := connection.recv()) is not None:
        if isinstance(message, Restart):
            frame, _ = env.env.reset()
            _, info = env.start_from(message.state, message.emulator_state, frame)
            connection.send((practice_view(env), info["state"]))
            continue
        state = info["state"]
        _, reward, terminated, truncated, info = env.step(message)
        reached = info["state"]
        ended = terminated or truncated
        if ended:
            _, info = env.reset()
        step = PracticeStep(
            practice_view(env), reward, terminated or reached != state, truncated, reached, ended, info["state"]
        )
        connection.send(step)
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
        self.views: list[PracticeView] = [view for view, _ in starts]
        self.states: list[str] = [state for _, state in starts]

    @property
    def inputs(self) -> list[np.ndarray]:
        return [view.inputs for view in self.views]

    def send(self, actions: list[int]) -> None:
        for connection, action in zip(self.connections, actions, strict=True):
            connection.send(action)

    def restart(self, worker: int, state: str, emulator_state: bytes) -> None:
        self.connections[worker].send(Restart(state, emulator_state))
        self.views[worker], self.states[worker] = self.connections[worker].recv()

    def receive(self) -> list[PracticeStep]:
        steps: list[PracticeStep] = [connection.recv() for connection in self.connections]
        self.views = [step.view for step in steps]
        self.states = [step.state for step in steps]
        return steps

    def close(self) -> None:
        for connection in self.connections:
            connection.send(None)
        for process in self.processes:
            process.join(STOP_SECONDS)
            if process.is_alive():
                process.kill()
