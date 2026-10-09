import threading
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from pathlib import Path

from fakes import FakeWrapper, fake_environment, write_config

from datenwissenschaften.settings import load_config
from datenwissenschaften.training import practice
from datenwissenschaften.training.practice import PracticeEnvironments, PracticeStep, Restart

SCRIPT = [(3, 0), (3, 1), (0, 1)]
STEPS_TO_GAME_OVER = 2


class ThreadProcess:
    def __init__(self, target, args, daemon: bool) -> None:
        self.thread = threading.Thread(target=target, args=args, daemon=daemon)
        self.killed = False

    def start(self) -> None:
        self.thread.start()

    def join(self, timeout: float) -> None:
        self.thread.join(timeout)

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def kill(self) -> None:
        self.killed = True


class SharedEnd:
    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    def send(self, message: object) -> None:
        self.connection.send(message)

    def recv(self) -> object:
        return self.connection.recv()

    def close(self) -> None:
        pass


class ThreadContext:
    Process = ThreadProcess

    @staticmethod
    def Pipe() -> tuple[Connection, SharedEnd]:
        parent, child = Pipe()
        return parent, SharedEnd(child)


def practising(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(practice, "setup_logging", lambda level: None)
    monkeypatch.setattr(practice, "make_environment", lambda *args: fake_environment(tmp_path, SCRIPT))
    monkeypatch.setattr(practice.multiprocessing, "get_context", lambda method: ThreadContext)


def test_practice_workers_step_restart_and_stop(monkeypatch, tmp_path: Path):
    practising(monkeypatch, tmp_path)
    environments = PracticeEnvironments(FakeWrapper, load_config(write_config(tmp_path)), False, range(2))

    environments.send([0, 1])
    steps = environments.receive()
    environments.restart(1, "Survive", b"emulator")
    environments.close()

    assert environments.states == ["Survive", "Survive"]
    assert all(isinstance(step, PracticeStep) for step in steps)
    assert not any(process.is_alive() for process in environments.processes)


def test_a_practice_worker_starts_over_when_its_episode_ends(monkeypatch, tmp_path: Path):
    practising(monkeypatch, tmp_path)
    parent, child = Pipe()
    worker = threading.Thread(
        target=practice.practise, args=(FakeWrapper, load_config(write_config(tmp_path)), True, 0, child)
    )
    worker.start()

    parent.recv()
    steps = []
    for _ in range(STEPS_TO_GAME_OVER):
        parent.send(0)
        steps.append(parent.recv())
    parent.send(Restart("Survive", b"emulator"))
    restarted = parent.recv()
    parent.send(None)
    worker.join()

    assert steps[-1].ended
    assert steps[-1].terminal
    assert restarted[1] == "Survive"
