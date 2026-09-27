from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class EpisodeRecord:
    episode_index: int
    bk2_path: str
    started_from_initial_savestate: bool
    episode_start_state: str
    curriculum_state: str
    duration_seconds: float
    step_count: int
    score: float
    won: bool
    curriculum_succeeded: bool
    curriculum_mastered: bool
    final_state: str

    @classmethod
    def start(cls, episode_index: int, info: dict[str, Any]) -> "EpisodeRecord":
        return cls(
            episode_index=episode_index,
            bk2_path=info["episode_bk2_path"],
            started_from_initial_savestate=info["started_from_initial_savestate"],
            episode_start_state=info["episode_start_state"],
            curriculum_state=info["state"],
            duration_seconds=0.0,
            step_count=0,
            score=0.0,
            won=False,
            curriculum_succeeded=False,
            curriculum_mastered=False,
            final_state=info["state"],
        )

    def add_step(self, info: dict[str, Any], reward: float) -> None:
        self.step_count += 1
        self.score += reward
        self.won = self.won or info["won"]
        self.curriculum_state = info["curriculum_state"]
        self.curriculum_succeeded = self.curriculum_succeeded or info["curriculum_succeeded"]
        self.curriculum_mastered = self.curriculum_mastered or info["curriculum_mastered"]
        self.final_state = info["state"]
