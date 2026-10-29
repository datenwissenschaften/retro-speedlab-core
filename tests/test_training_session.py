from pathlib import Path

import numpy as np
import torch
from fakes import fake_decision, fake_environment, write_config

from datenwissenschaften.advisor.advice import Advice
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import IMITATION_WEIGHT, MASTERED_IMITATION_WEIGHT, DemonstrationStep, Lesson
from datenwissenschaften.settings import load_config
from datenwissenschaften.training import session as session_module
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.learning import Learning
from datenwissenschaften.training.lessons import Lessons
from datenwissenschaften.training.own_wins import OwnWins
from datenwissenschaften.training.practice import PracticeStep, PracticeView
from datenwissenschaften.training.session import (
    EXPLORATION_ONCE_MASTERED,
    EXPLORATION_WHILE_LEARNING,
    ROLLOUT_STEPS,
    TrainingSession,
)
from datenwissenschaften.training.state_models import StateModels

IDLE_LAB = LabRun(Path("no-lab-run"))
VIEW = PracticeView(np.zeros(2, dtype=np.float32), {}, "Which move survives?")


class FakeLayaPractice:
    def __init__(self, workers: int) -> None:
        self.states = ["Survive"] * workers
        self.sent: list[list[int]] = []

    def views(self) -> tuple[list[dict[str, str]], list[int]]:
        return [{"state": "{}", "question": VIEW.question}] * len(self.states), [0] * len(self.states)

    def send(self, decisions: list[Decision]) -> None:
        self.sent.append([decision.action for decision in decisions])

    def receive(self) -> list[PracticeStep]:
        return [PracticeStep(VIEW, 1.0, False, False, "Survive", False, "Survive") for _ in self.states]


class RecordingPolicy:
    def __init__(self) -> None:
        self.num_timesteps = 0


class RecordingAgent:
    def __init__(self) -> None:
        self.policy = RecordingPolicy()
        self.rollouts: list[int] = []
        self.explorations: list[float] = []
        self.demonstrations: list[int] = []
        self.advice: list[int | None] = []

    @property
    def num_timesteps(self) -> int:
        return self.policy.num_timesteps

    def restart(self) -> None:
        self.policy = RecordingPolicy()

    def read(self, observations: list[dict[str, str]]) -> tuple[torch.Tensor, torch.Tensor]:
        rows = torch.zeros(len(observations), 1)
        return rows, rows

    def decide(
        self, features: tuple[torch.Tensor, torch.Tensor], exploration: float, advice: list[int | None]
    ) -> list[Decision]:
        self.explorations.append(exploration)
        self.advice += advice
        return [fake_decision(1, {"left": 0.2, "right": 0.8}, 0.74) for _ in range(len(features[0]))]

    def learn(self, rollout, lesson) -> None:
        self.rollouts.append(len(rollout))
        self.demonstrations.append(len(lesson.examples))
        self.policy.num_timesteps += len(rollout)


class FakeCoach:
    def __init__(self) -> None:
        self.checks = 0

    def check(self) -> None:
        self.checks += 1


def lessons_of(env, demonstrations: dict[str, list[DemonstrationStep]]) -> Lessons:
    curriculum = env.curriculum.curriculum
    learning = Learning(curriculum, env.curriculum.drill)
    wins = OwnWins(curriculum.state_names, lambda state: curriculum.root / f"{state}.wins.json", learning)
    return Lessons(demonstrations, learning, wins)


def no_lessons(env) -> Lessons:
    return lessons_of(env, {})


class RecordingHook:
    def __init__(self) -> None:
        self.steps: list[Transition] = []
        self.episodes: list[EpisodeRecord] = []
        self.updates = 0

    def on_step(self, transition: Transition) -> None:
        self.steps.append(transition)

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        self.episodes.append(episode)

    def on_update(self) -> None:
        self.updates += 1


def test_session_plays_learns_and_stops_on_a_reset_request(tmp_path: Path, monkeypatch):
    requests = iter([None] * (ROLLOUT_STEPS + 1) + ["reset"])
    monkeypatch.setattr(session_module, "consume_model_reset", lambda: next(requests))
    env = fake_environment(tmp_path, [(3, 0), (3, 1), (0, 2)])
    agent, hook = RecordingAgent(), RecordingHook()
    models = StateModels(agent, RunContext(load_config(write_config(tmp_path)), "Level1"), ("Survive", "Boss"))

    result = TrainingSession(
        env, models, [hook], float("inf"), no_lessons(env), IDLE_LAB, FakeCoach(), FakeLayaPractice(0)
    ).run()

    assert result == "reset"
    assert agent.num_timesteps == ROLLOUT_STEPS
    assert hook.steps[-1].timesteps == ROLLOUT_STEPS + 1
    assert agent.rollouts == [ROLLOUT_STEPS]
    assert hook.updates == 1
    assert len(hook.steps) == ROLLOUT_STEPS + 1
    assert hook.steps[0].decision.action == 1
    assert len(hook.steps[0].frames) == env.action_table.shape[1]
    first = hook.episodes[0]
    assert (first.episode_index, first.step_count, first.score, first.final_state) == (0, 2, 3.0, "Survive")
    assert first.duration_seconds >= 0.0
    assert hook.episodes[1].episode_index == 1
    assert set(agent.explorations) == {EXPLORATION_WHILE_LEARNING}


def test_mastered_states_explore_less(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    models = StateModels(RecordingAgent(), context, ("Survive", "Boss"))
    for _ in range(env.curriculum.curriculum.WIN_TARGET):
        env.curriculum.curriculum.record_success("Survive", 1)

    session = TrainingSession(
        env, models, [], float("inf"), no_lessons(env), IDLE_LAB, FakeCoach(), FakeLayaPractice(0)
    )

    assert session._exploration("Survive") == EXPLORATION_ONCE_MASTERED
    assert session._exploration("Boss") == EXPLORATION_WHILE_LEARNING
    env.curriculum.drill.state = "Survive"
    assert session._exploration("Survive") == EXPLORATION_WHILE_LEARNING


def test_session_hands_over_to_the_next_level_after_an_episode_past_the_deadline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(session_module, "consume_model_reset", lambda: None)
    env = fake_environment(tmp_path, [(3, 0), (3, 1), (0, 2)])
    hook = RecordingHook()
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    models = StateModels(RecordingAgent(), context, ("Survive", "Boss"))

    result = TrainingSession(
        env, models, [hook], 0.0, no_lessons(env), IDLE_LAB, FakeCoach(), FakeLayaPractice(0)
    ).run()

    assert result is None
    assert len(hook.episodes) == 1


def test_mastered_states_keep_a_light_imitation_and_a_drilled_state_learns_in_full(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    step = DemonstrationStep(
        "{}", "Which move survives?", 1, np.zeros(1, dtype=np.uint8), np.zeros(1, dtype=np.float32)
    )
    for _ in range(env.curriculum.curriculum.WIN_TARGET):
        env.curriculum.curriculum.record_success("Survive", 1)

    lessons = lessons_of(env, {"Survive": [step], "Boss": [step]})

    assert lessons("Survive") == Lesson([step], MASTERED_IMITATION_WEIGHT)
    assert lessons("Boss") == Lesson([step], IMITATION_WEIGHT)
    assert no_lessons(env)("Boss") == Lesson([], IMITATION_WEIGHT)
    env.curriculum.drill.state = "Survive"
    assert lessons("Survive") == Lesson([step], IMITATION_WEIGHT)


def test_laya_decides_with_the_advice_it_read_and_the_coach_is_watched(tmp_path: Path, monkeypatch):
    requests = iter([None, None, "reset"])
    monkeypatch.setattr(session_module, "consume_model_reset", lambda: next(requests))
    env = fake_environment(tmp_path, [(3, 0)] * 4)
    env.observer.advisor = lambda state, inputs: Advice(1, {"left": 0.1, "right": 0.9})
    agent, coach = RecordingAgent(), FakeCoach()
    models = StateModels(agent, RunContext(load_config(write_config(tmp_path)), "Level1"), ("Survive", "Boss"))

    TrainingSession(env, models, [], float("inf"), no_lessons(env), IDLE_LAB, coach, FakeLayaPractice(0)).run()

    assert agent.advice == [1, 1]
    assert coach.checks == 2
    assert models.rollout("Survive", 0).decisions[0].advice is None


def test_laya_also_plays_its_practice_emulators_and_learns_from_each(tmp_path: Path, monkeypatch):
    requests = iter([None, None, "reset"])
    monkeypatch.setattr(session_module, "consume_model_reset", lambda: next(requests))
    env = fake_environment(tmp_path, [(3, 0)] * 4)
    agent, practice = RecordingAgent(), FakeLayaPractice(2)
    models = StateModels(agent, RunContext(load_config(write_config(tmp_path)), "Level1"), ("Survive", "Boss"))

    TrainingSession(env, models, [], float("inf"), no_lessons(env), IDLE_LAB, FakeCoach(), practice).run()

    assert practice.sent == [[1, 1], [1, 1]]
    assert agent.advice == [None, 0, 0, None, 0, 0]
    assert [len(models.rollout("Survive", worker)) for worker in (0, 1, 2)] == [2, 2, 2]
    assert models.collected("Survive") == 6
