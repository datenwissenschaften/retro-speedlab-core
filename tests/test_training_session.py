from pathlib import Path

from fakes import fake_environment

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.training import session as session_module
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.session import ROLLOUT_STEPS, TrainingSession


class RecordingAgent:
    def __init__(self) -> None:
        self.num_timesteps = 0
        self.rollouts: list[int] = []

    def act(self, observation: dict[str, str]) -> Decision:
        return Decision(1, {"left": 0.2, "right": 0.8}, 0.74)

    def learn(self, rollout) -> None:
        self.rollouts.append(len(rollout))


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

    result = TrainingSession(env, agent, [hook]).run()

    assert result == "reset"
    assert agent.num_timesteps == ROLLOUT_STEPS + 1
    assert agent.rollouts == [ROLLOUT_STEPS]
    assert hook.updates == 1
    assert len(hook.steps) == ROLLOUT_STEPS + 1
    assert hook.steps[0].decision.action == 1
    assert len(hook.steps[0].frames) == env.action_table.shape[1]
    first = hook.episodes[0]
    assert (first.episode_index, first.step_count, first.score, first.final_state) == (0, 2, 3.0, "Survive")
    assert first.duration_seconds >= 0.0
    assert hook.episodes[1].episode_index == 1
