import threading
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from datenwissenschaften.advisor.advice import AdvisorDecision
from datenwissenschaften.advisor.advisors import Advisors
from datenwissenschaften.advisor.backplay import Backplay
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.rollout import Rollout
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.practice import PracticeEnvironments, PracticeStep
from datenwissenschaften.ui.telemetry import publish_metadata

ROLLOUT_STEPS = 2048
PAUSE_SECONDS = 5.0
REPORTED = ("num_timesteps", "entropy", "explained_variance", "imitation_loss")


class Coach:
    def __init__(
        self,
        advisors: Advisors,
        practice: PracticeEnvironments,
        lab_run: LabRun,
        lessons: Callable[[str], list[DemonstrationStep]],
        backplay: Backplay,
    ) -> None:
        self.advisors = advisors
        self.backplay = backplay
        self.backplayed: dict[int, str] = {}
        self.practice = practice
        self.lab_run = lab_run
        self.lessons = lessons
        self.rollouts: dict[tuple[str, int], Rollout[AdvisorDecision]] = {}
        self.exits: dict[str, Counter[str]] = {}
        self.stopping = threading.Event()
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="advisor-coach")
        self.running: Future[None] = self.worker.submit(self._coach)

    def check(self) -> None:
        if self.running.done():
            self.running.result()
            raise RuntimeError("The advisor coach stopped on its own.")

    def stop(self) -> None:
        self.stopping.set()
        self.worker.shutdown(wait=True)
        self.running.result()

    def _coach(self) -> None:
        while not self.stopping.is_set():
            if self.lab_run.deadline() is not None:
                self.stopping.wait(PAUSE_SECONDS)
                continue
            states = self.practice.states
            decisions = self.advisors.act(states, self.practice.inputs)
            self.practice.send([decision.action for decision in decisions])
            steps = self.practice.receive()
            for worker, (state, decision, step) in enumerate(zip(states, decisions, steps, strict=True)):
                rollout = self.rollouts.setdefault((state, worker), Rollout())
                rollout.add(decision, step.reward, step.terminal, step.truncated)
                if step.reached != state:
                    self.exits.setdefault(state, Counter())[step.reached] += 1
                self._follow_backplay(worker, step)
            for state in dict.fromkeys(states):
                self._learn(state)

    def _follow_backplay(self, worker: int, step: PracticeStep) -> None:
        if worker in self.backplayed:
            origin = self.backplayed[worker]
            if self.backplay.forward(origin, step.reached) or step.ended:
                self.backplay.record(origin, self.backplay.forward(origin, step.reached))
                del self.backplayed[worker]
        if not step.ended or (start := self.backplay.start(step.state)) is None:
            return
        self.practice.restart(worker, step.state, start)
        self.backplayed[worker] = step.state

    def _learn(self, state: str) -> None:
        keys = [key for key in self.rollouts if key[0] == state]
        if sum(len(self.rollouts[key]) for key in keys) < ROLLOUT_STEPS:
            return
        rollout = Rollout.joined([self.rollouts.pop(key) for key in keys])
        metrics = self.advisors.learn(state, rollout, self.lessons(state))
        report: dict[str, object] = {name: round(metrics[name], 3) for name in REPORTED}
        report["exits"] = dict(self.exits[state]) if state in self.exits else {}
        if state in self.backplay.starts:
            report["backplay"] = self.backplay.progress(state)
        publish_metadata("advisors", {state: report})
