import time

from datenwissenschaften.advisor.coach import Coach
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import TrainingHook, Transition
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.laya_practice import LayaPractice
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
        laya_practice: LayaPractice,
    ) -> None:
        self.env = env
        self.laya_practice = laya_practice
        self.deadline = deadline
        self.models = models
        self.hooks = hooks
        self.lessons = lessons
        self.lab_run = lab_run
        self.coach = coach
        self.episodes = 0

    def run(self) -> ModelResetRequest | None:
        observation, info = self.env.reset()
        episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
        while (request := consume_model_reset()) is None:
            if (request := self.lab_run.wait()) is not None:
                return request
            self.coach.check()
            state_name, advice = info["state"], info["advice"]
            practice_observations, practice_advice = self.laya_practice.views()
            states = [state_name, *self.laya_practice.states]
            exploration = {state: self._exploration(state) for state in states}
            observations = [observation, *practice_observations]
            decision, *practice = self.models.decide(observations, states, exploration, [advice, *practice_advice])
            self.laya_practice.send(practice)
            next_observation, reward, terminated, truncated, info = self.env.step(decision.action)
            self._practise(states[1:], practice)
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
                next_observation, info = self.env.reset()
                episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
            observation = next_observation
            for state in dict.fromkeys(states):
                self._learn(state)
        return request

    def _practise(self, states: list[str], decisions: list[Decision]) -> None:
        steps = self.laya_practice.receive()
        for worker, (state, decision, step) in enumerate(
            zip(states, decisions, steps, strict=True), MAIN_ENVIRONMENT + 1
        ):
            self.models.rollout(state, worker).add(decision, step.reward, step.terminal, step.truncated)

    def _learn(self, state_name: str) -> None:
        if self.models.collected(state_name) < ROLLOUT_STEPS:
            return
        self.models.learn(state_name, self.lessons(state_name))
        for hook in self.hooks:
            hook.on_update()

    def _exploration(self, state_name: str) -> float:
        mastered = self.env.curriculum.curriculum.is_mastered(state_name)
        return EXPLORATION_ONCE_MASTERED if mastered else EXPLORATION_WHILE_LEARNING
