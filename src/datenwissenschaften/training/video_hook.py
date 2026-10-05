import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger

from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.video_render import render_video

METADATA_SUFFIX = ".rollout.json"


class BestVideoHook:
    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.episodes: list[EpisodeRecord] = []
        self.updates = 0

    def on_step(self, transition: Transition) -> None:
        pass

    def on_episode_end(self, episode: EpisodeRecord) -> None:
        self.episodes.append(episode)

    def on_update(self) -> None:
        self.updates += 1
        best_by_curriculum: dict[str, EpisodeRecord] = {}
        for episode in self.episodes:
            incumbent = best_by_curriculum.get(episode.curriculum_state)
            if incumbent is None or (episode.score, -episode.step_count) > (incumbent.score, -incumbent.step_count):
                best_by_curriculum[episode.curriculum_state] = episode
        for episode in best_by_curriculum.values():
            self._record_if_best(episode)
        self.episodes.clear()

    def _record_if_best(self, episode: EpisodeRecord) -> None:
        previous_best = self._best_recorded_score(episode.curriculum_state)
        if previous_best is not None and episode.score <= previous_best:
            return
        if not Path(episode.bk2_path).is_file():
            logger.warning(f"Cannot render rollout video; recording is missing: {episode.bk2_path}")
            return
        source = Path(episode.bk2_path)
        video = source.with_suffix(".mp4")
        metadata_path = video.with_suffix(METADATA_SUFFIX)
        try:
            video.unlink(missing_ok=True)
            render_video(self.context.config.paths, source)
            metadata_path.write_text(json.dumps(self._metadata(episode, source, video)), encoding="utf-8")
            logger.info(f"Recorded best episode for {episode.curriculum_state}: score={episode.score:g}")
        except subprocess.CalledProcessError as error:
            video.unlink(missing_ok=True)
            logger.warning(f"Could not render rollout video from {source.name}: {error.stderr.strip()}")

    def _metadata(self, episode: EpisodeRecord, source: Path, video: Path) -> dict[str, object]:
        return {
            "game": self.context.game,
            "savestate": self.context.savestate,
            "curriculum": episode.curriculum_state,
            "episode_start_state": episode.episode_start_state,
            "started_from_initial_savestate": episode.started_from_initial_savestate,
            "full_run": episode.started_from_initial_savestate,
            "episode": episode.episode_index,
            "recording": source.name,
            "rollout": self.updates,
            "score": episode.score,
            "steps": episode.step_count,
            "won": episode.won,
            "curriculum_succeeded": episode.curriculum_succeeded,
            "curriculum_mastered": episode.curriculum_mastered,
            "video": video.name,
            "recorded_at": datetime.now(UTC).isoformat(),
        }

    def _best_recorded_score(self, curriculum: str) -> float | None:
        scores = []
        for metadata_path in self.context.record_dir.glob(f"*{METADATA_SUFFIX}"):
            video = metadata_path.with_name(metadata_path.name.removesuffix(METADATA_SUFFIX) + ".mp4")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if video.is_file() and metadata["curriculum"] == curriculum and math.isfinite(metadata["score"]):
                scores.append(float(metadata["score"]))
        return max(scores) if scores else None
