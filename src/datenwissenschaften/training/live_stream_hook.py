import json
from collections import deque

import cv2
import numpy as np

from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.story_teller import StoryTeller
from datenwissenschaften.ui.live import live_feed
from datenwissenschaften.ui.telemetry import best_fitness, episode_count, level_episode_count
from datenwissenschaften.vision.overlay import draw_detections

JPEG_QUALITY = 80
RECENT_SCORES = 120


class LiveStreamHook:
    def __init__(self, frame_rate: float, teller: StoryTeller, savestate: str) -> None:
        self.frame_rate = frame_rate
        self.teller = teller
        self.savestate = savestate
        self.episode = episode_count() + 1
        self.attempt = level_episode_count(savestate) + 1
        self.episode_reward = 0.0
        self.updates = 0
        self.recent_scores: deque[float] = deque(maxlen=RECENT_SCORES)
        self.step = 0

    def on_step(self, transition: Transition) -> None:
        self.episode_reward += transition.reward
        self.teller.observe(transition, self.attempt)
        self.step += 1
        status = {
            "step": self.step,
            "timesteps": transition.timesteps,
            "episode": self.episode,
            "attempt": self.attempt,
            "level": self.savestate,
            "episode_reward": self.episode_reward,
            "updates": self.updates,
            "training_state": transition.info["state"],
            "action": list(transition.decision.probabilities)[transition.decision.action],
            "probabilities": transition.decision.probabilities,
            "question": transition.observation["question"],
            "ram": json.loads(transition.observation["state"]),
        }
        detections = transition.info["detections"]
        for frame in transition.frames:
            live_feed.record(self._jpeg(draw_detections(frame, detections)), status)

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        self.recent_scores.append(episode.score)
        previous_best = best_fitness(self.savestate)
        new_best = previous_best is not None and episode.score > previous_best
        self.teller.finish(episode, live_feed.last_image())
        result = {
            "score": episode.score,
            "won": episode.won,
            "new_best": new_best,
            "full_run": episode.started_from_initial_savestate,
            "succeeded": episode.curriculum_succeeded or episode.won,
            "attempt": self.attempt,
            "level": self.savestate,
        }
        live_feed.finish_episode(self.episode, self.frame_rate, result, {"recent_scores": list(self.recent_scores)})
        self.episode += 1
        self.attempt += 1
        self.episode_reward = 0.0

    def on_update(self) -> None:
        self.updates += 1

    @staticmethod
    def _jpeg(frame: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
