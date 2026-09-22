from pathlib import Path
from types import SimpleNamespace

from datenwissenschaften.callbacks import upload_episode_callback as upload_episode_callback_module
from datenwissenschaften.callbacks.episode_record import EpisodeRecord
from datenwissenschaften.callbacks.upload_episode_callback import (
    UploadEpisodeCallback,
    _get_cpu_name,
    _get_total_memory_bytes,
    _parse_nvidia_smi_gpu,
    get_system_metadata,
)
from datenwissenschaften.settings import UploadSettings


class _FakeResponse:
    def __init__(self, payload=None, *, raises=False):
        self._payload = payload
        self._raises = raises

    def raise_for_status(self):
        if self._raises:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


def test_get_system_metadata_returns_expected_shape():
    metadata = get_system_metadata()

    assert "platform" in metadata
    assert "cpu" in metadata
    assert "memory" in metadata
    assert "gpu" in metadata


def test_get_cpu_name_falls_back_to_processor_on_non_linux(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(upload_episode_callback_module.platform, "processor", lambda: "arm64")

    assert _get_cpu_name() == "arm64"


def test_get_cpu_name_returns_none_when_no_processor_info(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(upload_episode_callback_module.platform, "processor", lambda: "")

    assert _get_cpu_name() is None


def test_get_cpu_name_returns_none_when_proc_cpuinfo_is_unreadable(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.platform, "system", lambda: "Linux")
    monkeypatch.setattr(upload_episode_callback_module.platform, "processor", lambda: "")

    def _raising_open(*_args, **_kwargs):
        raise OSError("no cpuinfo")

    monkeypatch.setattr("builtins.open", _raising_open)

    assert _get_cpu_name() is None


def test_get_total_memory_bytes_returns_none_when_sysconf_unavailable(monkeypatch):
    def _raise(name):
        raise ValueError("unknown sysconf name")

    monkeypatch.setattr(upload_episode_callback_module.os, "sysconf", _raise)

    assert _get_total_memory_bytes() is None


def test_get_gpu_metadata_records_torch_error(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.shutil, "which", lambda name: None)
    import torch

    def _raise_is_available():
        raise RuntimeError("cuda probe failed")

    monkeypatch.setattr(torch.cuda, "is_available", _raise_is_available)

    metadata = get_system_metadata()["gpu"]

    assert "torch_error" in metadata


def test_get_gpu_metadata_lists_cuda_devices(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.shutil, "which", lambda name: None)
    import torch

    fake_properties = SimpleNamespace(name="Fake GPU", total_memory=1024, major=8, minor=6, multi_processor_count=16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: fake_properties)

    metadata = get_system_metadata()["gpu"]

    assert metadata["cuda_available"] is True
    assert metadata["devices"] == [
        {
            "index": 0,
            "name": "Fake GPU",
            "total_memory_bytes": 1024,
            "major": 8,
            "minor": 6,
            "multi_processor_count": 16,
        }
    ]


def test_get_gpu_metadata_parses_nvidia_smi_output(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(
        upload_episode_callback_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="GeForce RTX 4090, 24576, 535.104.05\n"),
    )

    metadata = get_system_metadata()["gpu"]

    assert metadata["nvidia_smi"] == [
        {"name": "GeForce RTX 4090", "memory_total_mb": 24576, "driver_version": "535.104.05"}
    ]


def test_get_gpu_metadata_records_nvidia_smi_error(monkeypatch):
    monkeypatch.setattr(upload_episode_callback_module.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _raise(*_args, **_kwargs):
        raise OSError("nvidia-smi crashed")

    monkeypatch.setattr(upload_episode_callback_module.subprocess, "run", _raise)

    metadata = get_system_metadata()["gpu"]

    assert "nvidia_smi_error" in metadata


def test_parse_nvidia_smi_gpu_line():
    parsed = _parse_nvidia_smi_gpu("GeForce RTX 4090, 24576, 535.104.05")

    assert parsed == {"name": "GeForce RTX 4090", "memory_total_mb": 24576, "driver_version": "535.104.05"}


def test_upload_settings_default_from_config_when_not_provided(monkeypatch):
    fake_config = SimpleNamespace(upload=UploadSettings(url="https://cfg.test", api_key=None))
    monkeypatch.setattr(upload_episode_callback_module, "load_config", lambda path: fake_config)

    callback = UploadEpisodeCallback()

    assert callback.upload_url == "https://cfg.test"


def test_resolve_episode_path_falls_back_to_glob_search(tmp_path: Path):
    nested = tmp_path / "Weird" / "Layout" / "3" / "clip.bk2"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"movie")
    runtime = SimpleNamespace(record_dir=tmp_path, game="Game", savestate="Level1")

    resolved = UploadEpisodeCallback._resolve_episode_path("somewhere/3/clip.bk2", runtime)

    assert resolved == str(nested)


def test_resolve_episode_path_returns_none_when_nothing_matches(tmp_path: Path):
    runtime = SimpleNamespace(record_dir=tmp_path, game="Game", savestate="Level1")

    assert UploadEpisodeCallback._resolve_episode_path("missing/9/clip.bk2", runtime) is None


def test_on_step_returns_true_when_locals_are_incomplete():
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    callback.locals = {"rewards": None, "dones": None, "infos": None}

    assert callback._on_step() is True


def test_on_step_accumulates_steps_and_finishes_done_episodes(monkeypatch):
    runtime = SimpleNamespace(game="Game", savestate="Level1", record_dir="/records")
    monkeypatch.setattr(upload_episode_callback_module, "get_runtime", lambda: runtime)
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    callback.locals = {
        "rewards": [1.0, 2.0],
        "dones": [False, True],
        "infos": [{"won": False}, {"won": True}],
    }

    result = callback._on_step()

    assert result is True
    assert len(callback.completed_episodes) == 1
    assert callback.active_episodes[0].step_count == 1
    assert callback.active_episodes[1].step_count == 0


def test_finish_episode_builds_path_from_runtime_when_missing(monkeypatch):
    runtime = SimpleNamespace(game="Game", savestate="Level1", record_dir="/records")
    monkeypatch.setattr(upload_episode_callback_module, "get_runtime", lambda: runtime)
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    callback._ensure_episode_slots(1)
    episode = EpisodeRecord(0, 3)

    callback._finish_episode(0, episode)

    assert callback.completed_episodes[0].bk2_path == str(Path("/records/Game/Level1/0/Game-Level1-000003.bk2"))


def test_on_rollout_end_flushes_even_with_no_completed_episodes(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))
    flushed = []
    monkeypatch.setattr(callback, "_flush_successful_episodes", lambda: flushed.append(True))

    result = callback._on_rollout_end()

    assert result is True
    assert flushed == [True]


def test_flush_successful_episodes_returns_early_when_nothing_to_upload(monkeypatch):
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="key"))

    def _boom(*_args, **_kwargs):
        raise AssertionError("requests.get should not be called")

    monkeypatch.setattr(upload_episode_callback_module.requests, "get", _boom)

    callback._flush_successful_episodes()


def test_flush_successful_episodes_skips_upload_when_api_key_missing(monkeypatch, tmp_path: Path):
    episode_path = tmp_path / "episode.bk2"
    episode_path.write_bytes(b"movie")
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key=None))
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = str(episode_path)
    callback.successful_episodes = [episode]
    callback.model = SimpleNamespace(num_timesteps=1)
    monkeypatch.setattr(
        upload_episode_callback_module,
        "get_runtime",
        lambda: SimpleNamespace(
            get_model_metadata=lambda model: {},
            game="Game",
            savestate="Level1",
            record_dir=str(tmp_path),
        ),
    )

    callback._flush_successful_episodes()

    assert callback.successful_episodes == []


def test_flush_successful_episodes_logs_error_when_signing_key_request_fails(monkeypatch, tmp_path: Path):
    episode_path = tmp_path / "episode.bk2"
    episode_path.write_bytes(b"movie")
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="secret"))
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = str(episode_path)
    callback.successful_episodes = [episode]
    callback.model = SimpleNamespace(num_timesteps=1)
    monkeypatch.setattr(
        upload_episode_callback_module,
        "get_runtime",
        lambda: SimpleNamespace(
            get_model_metadata=lambda model: {},
            game="Game",
            savestate="Level1",
            record_dir=str(tmp_path),
        ),
    )

    def _raise(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(upload_episode_callback_module.requests, "get", _raise)

    callback._flush_successful_episodes()

    assert callback.successful_episodes == [episode]


def test_flush_successful_episodes_uploads_and_removes_confirmed_episodes(monkeypatch, tmp_path: Path):
    episode_path = tmp_path / "episode.bk2"
    episode_path.write_bytes(b"movie-bytes")
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="secret"))
    episode = EpisodeRecord(0, 0)
    episode.bk2_path = str(episode_path)
    callback.successful_episodes = [episode]
    callback.model = SimpleNamespace(num_timesteps=10)
    callback.num_timesteps = 10
    runtime = SimpleNamespace(
        get_model_metadata=lambda model: {"num_timesteps": model.num_timesteps},
        game="Game",
        savestate="Level1",
        record_dir=str(tmp_path),
    )
    monkeypatch.setattr(upload_episode_callback_module, "get_runtime", lambda: runtime)
    monkeypatch.setattr(
        upload_episode_callback_module.requests,
        "get",
        lambda url, headers, timeout: _FakeResponse({"signing_key": "abc123"}),
    )
    posted = []

    def _post(url, files, data, headers, timeout):
        posted.append((url, data, headers))
        return _FakeResponse()

    monkeypatch.setattr(upload_episode_callback_module.requests, "post", _post)

    callback._flush_successful_episodes()

    assert callback.successful_episodes == []
    assert len(posted) == 1
    assert posted[0][1] == {"game": "Game", "category": "Level1"}


def test_upload_episode_returns_false_and_logs_on_failure(monkeypatch, tmp_path: Path):
    episode_path = tmp_path / "episode.bk2"
    episode_path.write_bytes(b"movie-bytes")
    callback = UploadEpisodeCallback(UploadSettings(url="https://example.test", api_key="secret"))
    runtime = SimpleNamespace(game="Game", savestate="Level1")

    def _raise(*_args, **_kwargs):
        raise RuntimeError("upload failed")

    monkeypatch.setattr(upload_episode_callback_module.requests, "post", _raise)

    result = callback._upload_episode(str(episode_path), b"signed", "secret", runtime)

    assert result is False
