import time

from datenwissenschaften.environment.demonstration import Demonstrations
from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.laya.imitation import DemonstrationStep
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import TrainingHook, Transition
from datenwissenschaften.training.lab_run import LabRun
from datenwissenschaften.training.practice import PracticeEnvironments
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset

ROLLOUT_STEPS = 256
EXPLORATION_WHILE_LEARNING = 0.2
EXPLORATION_ONCE_MASTERED = 0.05
MAIN_ENVIRONMENT = 0


class TrainingSession:
    def __init__(
        self,
        env: StateMachineGymWrapper,
        models: StateModels,
        hooks: list[TrainingHook],
        deadline: float,
        demonstrations: Demonstrations,
        lab_run: LabRun,
        practice: PracticeEnvironments,
    ) -> None:
        self.env = env
        self.deadline = deadline
        self.models = models
        self.hooks = hooks
        self.demonstrations = demonstrations
        self.lab_run = lab_run
        self.practice = practice
        self.episodes = 0

    def run(self) -> ModelResetRequest | None:
        if (request := self.lab_run.wait()) is not None:
            return request
        observation, info = self.env.reset()
        episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
        while (request := consume_model_reset()) is None:
            state_name = info["state"]
            states = [state_name, *self.practice.states]
            explorations = {state: self._exploration(state) for state in states}
            decisions = self.models.decide([observation, *self.practice.observations], states, explorations)
            self.practice.send([decision.action for decision in decisions[1:]])
            decision = decisions[MAIN_ENVIRONMENT]
            next_observation, reward, terminated, truncated, info = self.env.step(decision.action)
            done = terminated or truncated
            self.models.rollout(state_name, MAIN_ENVIRONMENT).add(decision, reward, done or info["state"] != state_name)
            episode.add_step(info, reward)
            timesteps = self.models.timesteps(state_name)
            transition = Transition(timesteps, observation, decision, self.env.frames, reward, done, info)
            for hook in self.hooks:
                hook.on_step(transition)
            self._practise(states[1:], decisions[1:])
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
            self._learn(states)
        return request

    def _practise(self, states: list[str], decisions: list[Decision]) -> None:
        steps = self.practice.receive()
        for worker, (state, decision, step) in enumerate(zip(states, decisions, steps, strict=True), start=1):
            self.models.rollout(state, worker).add(decision, step.reward, step.segment_ends)

    def _learn(self, states: list[str]) -> None:
        for state_name in dict.fromkeys(states):
            if self.models.collected(state_name) < ROLLOUT_STEPS:
                continue
            self.models.learn(state_name, self._demonstrations(state_name))
            for hook in self.hooks:
                hook.on_update()

    def _demonstrations(self, state_name: str) -> list[DemonstrationStep]:
        if state_name not in self.demonstrations or self.env.curriculum.curriculum.is_mastered(state_name):
            return []
        return self.demonstrations[state_name]

    def _exploration(self, state_name: str) -> float:
        mastered = self.env.curriculum.curriculum.is_mastered(state_name)
        return EXPLORATION_ONCE_MASTERED if mastered else EXPLORATION_WHILE_LEARNING
