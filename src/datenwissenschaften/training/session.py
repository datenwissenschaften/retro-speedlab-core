import time

from datenwissenschaften.advisor.coach import Coach
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import TrainingHook, Transition
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.lessons import Lessons
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset
from datenwissenschaften.ui.telemetry import publish_metadata

ROLLOUT_STEPS = 256
EXPLORATION_WHILE_LEARNING = 0.05
EXPLORATION_ONCE_MASTERED = 0.01
MAIN_ENVIRONMENT = 0


class TrainingSession:
    def __init__(
        self,
        env: StateMachineGymWrapper,
        models: StateModels,
        hooks: list[TrainingHook],
        deadline: float,
        lessons: Lessons,
        lab_run: LabRun,
        coach: Coach,
    ) -> None:
        self.env = env
        self.deadline = deadline
        self.models = models
        self.hooks = hooks
        self.lessons = lessons
        self.lab_run = lab_run
        self.coach = coach
        self.episodes = 0

    def run(self) -> ModelResetRequest | None:
        if (request := self.lab_run.wait()) is not None:
            return request
        observation, info = self.env.reset()
        episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
        while (request := consume_model_reset()) is None:
            self.coach.check()
            state_name, advice = info["state"], info["advice"]
            exploration = {state_name: self._exploration(state_name)}
            decision = self.models.decide([observation], [state_name], exploration, [advice])[0]
            next_observation, reward, terminated, truncated, info = self.env.step(decision.action)
            done = terminated or truncated
            terminal = terminated or info["state"] != state_name
            self.models.rollout(state_name, MAIN_ENVIRONMENT).add(decision, reward, terminal, truncated)
            episode.add_step(info, reward)
            timesteps = self.models.timesteps(state_name)
            transition = Transition(timesteps, observation, decision, self.env.frames, reward, done, info)
            for hook in self.hooks:
                hook.on_step(transition)
            if done:
                episode.duration_seconds = time.monotonic() - started_at
                publish_metadata("routes", self.env.routes.summary(), replace=True)
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
            self._learn(state_name)
        return request

    def _learn(self, state_name: str) -> None:
        if self.models.collected(state_name) < ROLLOUT_STEPS:
            return
        self.models.learn(state_name, self.lessons(state_name))
        for hook in self.hooks:
            hook.on_update()

    def _exploration(self, state_name: str) -> float:
        mastered = self.env.curriculum.curriculum.is_mastered(state_name)
        return EXPLORATION_ONCE_MASTERED if mastered else EXPLORATION_WHILE_LEARNING
