import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fakes import write_config

from datenwissenschaften.laya.decision import Decision
from datenwissenschaften.settings import load_config
from datenwissenschaften.training import (
    checkpoint_hook,
    live_stream_hook,
    system,
    telemetry_hook,
    upload_hook,
    video_hook,
)
from datenwissenschaften.training.context import RunContext
from datenwissenschaften.training.episode_record import EpisodeRecord
from datenwissenschaften.training.hooks import Transition
from datenwissenschaften.training.state_models import StateModels
from datenwissenschaften.ui.live import LiveFeed
from datenwissenschaften.vision.detection import Detection

INFO = {
    "episode_bk2_path": "run.bk2",
    "started_from_initial_savestate": True,
    "episode_start_state": "Level1",
    "episode_start_score": 0.0,
    "state": "Survive",
}
OBSERVATION = {"state": '{"lives": 3}', "question": "Which move survives?"}
FRAMES = [np.zeros((4, 4, 3), np.uint8), np.ones((4, 4, 3), np.uint8)]


class FakeAgent:
    num_timesteps = 7

    def save(self, path: Path) -> None:
        path.write_bytes(b"weights")

    def metadata(self) -> dict[str, object]:
        return {"checkpoint": "fake/laya"}


@pytest.fixture
def context(tmp_path: Path) -> RunContext:
    return RunContext(load_config(write_config(tmp_path)))


def _episode(bk2_path: str, score: float, won: bool, full_run: bool) -> EpisodeRecord:
    episode = EpisodeRecord.start(3, {**INFO, "episode_bk2_path": bk2_path, "started_from_initial_savestate": full_run})
    episode.add_step(
        {
            "won": won,
            "curriculum_state": "Survive",
            "curriculum_succeeded": won,
            "curriculum_mastered": False,
            "state": "Survive",
        },
        score,
    )
    return episode


def _transition() -> Transition:
    info = {"state": "Survive", "detections": (Detection("door", 0, 0, 2, 2),)}
    return Transition(7, OBSERVATION, Decision(1, {"left": 0.3, "right": 0.7}, 0.66), FRAMES, 2.0, False, info)


def test_context_places_the_model_per_game_and_savestate(context: RunContext):
    assert (
        context.model_path("Survive")
        == context.config.paths.models_dir / "FakeGame-v0" / "Level1" / "Survive" / "laya.pt"
    )
    assert context.record_dir == context.config.paths.record_dir / "FakeGame-v0" / "Level1"


def test_telemetry_hook_publishes_finished_episodes(context: RunContext, monkeypatch):
    published = []
    monkeypatch.setattr(telemetry_hook, "publish_episode", lambda **values: published.append(values))
    hook = telemetry_hook.TelemetryHook(context)

    hook.on_step(_transition())
    hook.on_episode_end(_episode("run.bk2", 4.0, True, True))
    hook.on_update()

    assert published[0]["fitness"] == 4.0
    assert published[0]["won"] is True
    assert published[0]["savestate"] == "Level1"


def test_checkpoint_hook_saves_and_publishes_metadata(context: RunContext, monkeypatch):
    published = []
    monkeypatch.setattr(checkpoint_hook, "publish_metadata", lambda *args, **kwargs: published.append(args))
    models = StateModels(FakeAgent(), context, ("Survive",))
    models.activate("Survive")
    hook = checkpoint_hook.CheckpointHook(models)

    hook.on_step(_transition())
    hook.on_episode_end(_episode("run.bk2", 1.0, False, True))
    hook.on_update()

    assert context.model_path("Survive").read_bytes() == b"weights"
    assert published[0][1]["laya"] == {"state": "Survive", "checkpoint": "fake/laya"}


def test_live_stream_hook_records_every_frame_of_an_episode_with_its_result(monkeypatch):
    feed = LiveFeed()
    best_scores = iter([None, 3.0])
    monkeypatch.setattr(live_stream_hook, "live_feed", feed)
    monkeypatch.setattr(live_stream_hook, "best_fitness", lambda: next(best_scores))
    monkeypatch.setattr(live_stream_hook, "episode_count", lambda: 41)
    hook = live_stream_hook.LiveStreamHook(50.0)

    hook.on_step(_transition())
    hook.on_update()
    hook.on_episode_end(_episode("run.bk2", 5.0, False, True))
    hook.on_step(_transition())
    hook.on_episode_end(_episode("run.bk2", 4.0, True, True))

    first, second = (feed.episode_frames(episode_id, 0) for episode_id in (42, 43))
    status = first[0]["status"]
    assert (len(first), len(second)) == (2, 2)
    assert (status["action"], status["probabilities"]["right"]) == ("right", 0.7)
    assert (status["ram"], status["episode_reward"]) == ({"lives": 3}, 2.0)
    latest = feed.latest_episode()
    assert latest["episode"]["frame_rate"] == 50.0
    assert latest["episode"]["result"] == {"score": 4.0, "won": True, "new_best": True}
    assert feed._episodes[0]["result"]["new_best"] is False
    assert latest["summary"] == {"recent_scores": [5.0, 4.0]}
    assert hook.updates == 1


def test_video_hook_renders_only_new_best_runs(context: RunContext, monkeypatch):
    context.record_dir.mkdir(parents=True)
    recording = context.record_dir / "run.bk2"
    recording.write_bytes(b"movie")
    renders = []

    def render(command, **kwargs):
        renders.append(command)
        Path(command[-1]).with_suffix(".mp4").write_bytes(b"video")

    monkeypatch.setattr(video_hook.subprocess, "run", render)
    hook = video_hook.BestVideoHook(context)

    hook.on_step(_transition())
    hook.on_episode_end(_episode(str(recording), 3.0, False, True))
    hook.on_episode_end(_episode(str(recording), 5.0, False, True))
    hook.on_update()
    hook.on_episode_end(_episode(str(recording), 4.0, False, True))
    hook.on_update()

    assert len(renders) == 1
    metadata = json.loads((context.record_dir / "run.rollout.json").read_text(encoding="utf-8"))
    assert (metadata["score"], metadata["curriculum"], metadata["rollout"]) == (5.0, "Survive", 1)


def test_video_hook_skips_missing_recordings_and_survives_render_failures(context: RunContext, monkeypatch):
    context.record_dir.mkdir(parents=True)
    recording = context.record_dir / "broken.bk2"
    recording.write_bytes(b"movie")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="boom")

    monkeypatch.setattr(video_hook.subprocess, "run", fail)
    hook = video_hook.BestVideoHook(context)

    hook.on_episode_end(_episode(str(context.record_dir / "missing.bk2"), 1.0, False, True))
    hook.on_update()
    hook.on_episode_end(_episode(str(recording), 2.0, False, True))
    hook.on_update()

    assert not (context.record_dir / "broken.mp4").exists()


def _response(method: str, url: str, **kwargs) -> httpx.Response:
    return httpx.Response(200, json={"signing_key": "secret"}, request=httpx.Request(method, url))


def test_upload_hook_uploads_only_complete_winning_runs(context: RunContext, monkeypatch, tmp_path: Path):
    recording = tmp_path / "win.bk2"
    recording.write_bytes(b"movie")
    posts = []
    monkeypatch.setattr(upload_hook, "system_metadata", lambda: {"cpu": "fake"})
    monkeypatch.setattr(upload_hook.httpx, "get", lambda url, **kwargs: _response("GET", url))
    monkeypatch.setattr(
        upload_hook.httpx, "post", lambda url, **kwargs: posts.append(kwargs["data"]) or _response("POST", url)
    )
    hook = upload_hook.UploadHook(context, FakeAgent())
    hook.settings = SimpleNamespace(url="https://upload.test", api_key="key")

    hook.on_step(_transition())
    hook.on_episode_end(_episode(str(recording), 9.0, True, True))
    hook.on_episode_end(_episode(str(recording), 9.0, True, False))
    hook.on_episode_end(_episode(str(recording), 1.0, False, True))
    hook.on_update()

    assert posts == [{"game": "FakeGame-v0", "category": "Level1"}]
    assert hook.pending == []


def test_upload_hook_keeps_runs_when_the_server_fails(context: RunContext, monkeypatch, tmp_path: Path):
    def fail(url, **kwargs):
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(upload_hook.httpx, "get", fail)
    hook = upload_hook.UploadHook(context, FakeAgent())
    hook.settings = SimpleNamespace(url="https://upload.test", api_key="key")

    hook.on_episode_end(_episode(str(tmp_path / "win.bk2"), 9.0, True, True))
    hook.on_update()

    assert len(hook.pending) == 1


def test_upload_hook_discards_runs_without_an_api_key(context: RunContext):
    hook = upload_hook.UploadHook(context, FakeAgent())

    hook.on_update()
    hook.on_episode_end(_episode("win.bk2", 9.0, True, True))
    hook.on_update()

    assert hook.pending == []


def test_system_metadata_reports_hardware(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        system.subprocess, "run", lambda command, **kwargs: SimpleNamespace(stdout="Fake GPU, 6144, 550.1\n")
    )

    metadata = system.system_metadata()

    assert metadata["gpu"]["nvidia_smi"] == [{"name": "Fake GPU", "memory_total_mb": 6144, "driver_version": "550.1"}]
    assert metadata["memory"]["total_bytes"] > 0
    assert metadata["cpu"]["name"]


def test_system_metadata_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda name: None)
    monkeypatch.setattr(system, "CPU_INFO", Path("/missing/cpuinfo"))

    metadata = system.system_metadata()

    assert metadata["gpu"]["nvidia_smi"] == []
