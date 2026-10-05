import time

from datenwissenschaften.environment.demonstration import Demonstrations
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import TrainingHook, Transition
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset

ROLLOUT_STEPS = 64
EXPLORATION_WHILE_LEARNING = 0.2
EXPLORATION_ONCE_MASTERED = 0.05


class TrainingSession:
    def __init__(
        self,
        env: StateMachineGymWrapper,
        models: StateModels,
        hooks: list[TrainingHook],
        deadline: float,
        demonstrations: Demonstrations,
        lab_run: LabRun,
    ) -> None:
        self.env = env
        self.deadline = deadline
        self.models = models
        self.hooks = hooks
        self.demonstrations = demonstrations
        self.lab_run = lab_run
        self.episodes = 0

    def run(self) -> ModelResetRequest | None:
        if (request := self.lab_run.wait()) is not None:
            return request
        observation, info = self.env.reset()
        episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
        while (request := consume_model_reset()) is None:
            state_name = info["state"]
            self.models.activate(state_name)
            decision = self.models.agent.act(observation, self._exploration(state_name))
            next_observation, reward, terminated, truncated, info = self.env.step(decision.action)
            done = terminated or truncated
            segment_ends = done or info["state"] != state_name
            rollout = self.models.rollout
            rollout.add(observation["state"], observation["question"], decision, reward, segment_ends)
            episode.add_step(info, reward)
            timesteps = self.models.agent.num_timesteps + len(rollout)
            transition = Transition(timesteps, observation, decision, self.env.frames, reward, done, info)
            for hook in self.hooks:
                hook.on_step(transition)
            if done:
                episode.duration_seconds = time.monotonic() - started_at
                for hook in self.hooks:
                    hook.on_episode_end(episode)
                self.episodes += 1
                if time.monotonic() >= self.deadline:
                    return None
                if (request := self.lab_run.wait()) is not None:
                    return request
                next_observation, info = self.env.reset()
                episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
            observation = next_observation
            if len(rollout) >= ROLLOUT_STEPS:
                self.models.learn(self._demonstrations(self.models.require_active()))
                for hook in self.hooks:
                    hook.on_update()
        return request

    def _demonstrations(self, state_name: str) -> list[DemonstrationStep]:
        if state_name not in self.demonstrations or self.env.curriculum.curriculum.is_mastered(state_name):
            return []
        return self.demonstrations[state_name]

    def _exploration(self, state_name: str) -> float:
        mastered = self.env.curriculum.curriculum.is_mastered(state_name)
        return EXPLORATION_ONCE_MASTERED if mastered else EXPLORATION_WHILE_LEARNING
