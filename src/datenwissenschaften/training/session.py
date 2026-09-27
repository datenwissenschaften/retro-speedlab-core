import time

from datenwissenschaften.environment.wrapper import StateMachineGymWrapper
from datenwissenschaften.laya.agent import LayaAgent
from datenwissenschaften.laya.rollout import Rollout
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import TrainingHook, Transition
from datenwissenschaften.ui.control import ModelResetRequest, consume_model_reset

ROLLOUT_STEPS = 64


class TrainingSession:
    def __init__(self, env: StateMachineGymWrapper, agent: LayaAgent, hooks: list[TrainingHook]) -> None:
        self.env = env
        self.agent = agent
        self.hooks = hooks
        self.episodes = 0

    def run(self) -> ModelResetRequest:
        observation, info = self.env.reset()
        episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
        rollout = Rollout()
        while (request := consume_model_reset()) is None:
            decision = self.agent.act(observation)
            next_observation, reward, terminated, truncated, info = self.env.step(decision.action)
            done = terminated or truncated
            self.agent.num_timesteps += 1
            rollout.add(observation["state"], observation["question"], decision, reward, done)
            episode.add_step(info, reward)
            transition = Transition(
                self.agent.num_timesteps, observation, decision, self.env.frames, reward, done, info
            )
            for hook in self.hooks:
                hook.on_step(transition)
            if done:
                episode.duration_seconds = time.monotonic() - started_at
                for hook in self.hooks:
                    hook.on_episode_end(episode)
                self.episodes += 1
                next_observation, info = self.env.reset()
                episode, started_at = EpisodeRecord.start(self.episodes, info), time.monotonic()
            observation = next_observation
            if len(rollout) >= ROLLOUT_STEPS:
                self.agent.learn(rollout)
                for hook in self.hooks:
                    hook.on_update()
                rollout = Rollout()
        return request
