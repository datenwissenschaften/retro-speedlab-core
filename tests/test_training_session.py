from pathlib import Path

from fakes import fake_environment, write_config

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.settings import load_config
from datenwissenschaften.training import session as session_module
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.session import (
    EXPLORATION_ONCE_MASTERED,
    EXPLORATION_WHILE_LEARNING,
    ROLLOUT_STEPS,
    TrainingSession,
)
from datenwissenschaften.training.state_models import StateModels


class RecordingAgent:
    def __init__(self) -> None:
        self.num_timesteps = 0
        self.rollouts: list[int] = []
        self.explorations: list[float] = []
        self.demonstrations: list[int] = []

    def act(self, observation: dict[str, str], exploration: float) -> Decision:
        self.explorations.append(exploration)
        return Decision(1, {"left": 0.2, "right": 0.8}, 0.74)

    def learn(self, rollout, demonstrations) -> None:
        self.rollouts.append(len(rollout))
        self.demonstrations.append(len(demonstrations))
        self.num_timesteps += len(rollout)


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

    result = TrainingSession(env, models, [hook], float("inf"), {}).run()

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

    session = TrainingSession(env, models, [], float("inf"), {})

    assert session._exploration("Survive") == EXPLORATION_ONCE_MASTERED
    assert session._exploration("Boss") == EXPLORATION_WHILE_LEARNING


def test_session_hands_over_to_the_next_level_after_an_episode_past_the_deadline(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(session_module, "consume_model_reset", lambda: None)
    env = fake_environment(tmp_path, [(3, 0), (3, 1), (0, 2)])
    hook = RecordingHook()
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    models = StateModels(RecordingAgent(), context, ("Survive", "Boss"))

    result = TrainingSession(env, models, [hook], 0.0, {}).run()

    assert result is None
    assert len(hook.episodes) == 1


def test_only_unmastered_states_learn_from_their_demonstrations(tmp_path: Path):
    env = fake_environment(tmp_path, [(3, 0)])
    context = RunContext(load_config(write_config(tmp_path)), "Level1")
    models = StateModels(RecordingAgent(), context, ("Survive", "Boss"))
    step = DemonstrationStep("{}", "Which move survives?", 1)
    for _ in range(env.curriculum.curriculum.WIN_TARGET):
        env.curriculum.curriculum.record_success("Survive", 1)

    session = TrainingSession(env, models, [], float("inf"), {"Survive": [step], "Boss": [step]})

    assert session._demonstrations("Survive") == []
    assert session._demonstrations("Boss") == [step]
    assert TrainingSession(env, models, [], float("inf"), {})._demonstrations("Boss") == []
