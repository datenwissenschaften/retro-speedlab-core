import json
from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften import rollout_video
from datenwissenschaften.callbacks.episode_record import EpisodeRecord


def _episode(path: Path, curriculum: str, score: float, *, env_index: int = 0) -> EpisodeRecord:
    episode = EpisodeRecord(env_index, 0)
    episode.bk2_path = str(path)
    episode.curriculum_state = curriculum
    episode.score = score
    episode.step_count = 10
    return episode


def test_missing_or_foreign_recording_is_skipped_with_a_warning(monkeypatch, tmp_path: Path):
    runtime = SimpleNamespace(
        record_dir=tmp_path,
        savestate="Level1",
        game="Game",
        paths=SimpleNamespace(roms_path=tmp_path / "roms"),
    )
    monkeypatch.setattr(rollout_video, "get_runtime", lambda: runtime)
    monkeypatch.setattr(rollout_video.subprocess, "run", lambda *args, **kwargs: None)

    videos = rollout_video.record_rollout_videos(
        [_episode(tmp_path / "nowhere" / "missing.bk2", "Explore", 5.0)],
        rollout=1,
    )

    assert videos == []


def test_best_recorded_score_ignores_metadata_without_a_video_file(tmp_path: Path):
    orphan_metadata = tmp_path / "Game" / "Level1" / "orphan.rollout.json"
    orphan_metadata.parent.mkdir(parents=True)
    orphan_metadata.write_text(json.dumps({"curriculum": "Explore", "score": 999.0}), encoding="utf-8")

    assert (
        rollout_video._best_recorded_score(
            tmp_path,
            game="Game",
            savestate="Level1",
            curriculum="Explore",
        )
        is None
    )


def test_resolve_recording_returns_none_for_an_empty_path(tmp_path: Path):
    assert (
        rollout_video._resolve_recording(
            "",
            tmp_path,
            game="Game",
            savestate="Level1",
            env_index=0,
        )
        is None
    )


def test_best_recorded_score_ignores_entries_for_a_different_curriculum(tmp_path: Path):
    other_dir = tmp_path / "Game" / "Level1"
    other_dir.mkdir(parents=True)
    metadata = other_dir / "fight.rollout.json"
    metadata.write_text(json.dumps({"curriculum": "Fight", "score": 999.0}), encoding="utf-8")
    (other_dir / "fight.mp4").write_bytes(b"video")

    assert (
        rollout_video._best_recorded_score(
            tmp_path,
            game="Game",
            savestate="Level1",
            curriculum="Explore",
        )
        is None
    )
