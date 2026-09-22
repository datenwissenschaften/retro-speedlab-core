import http.client
import json
import socket
from contextlib import closing, contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from datenwissenschaften.settings import UISettings
from datenwissenschaften.ui import server as server_module
from datenwissenschaften.ui.server import (
    DashboardServer,
    _redact_config_secrets,
    _source_language,
    generated_source,
    generated_sources,
    rollout_video_path,
    rollout_videos,
)


def _ui_settings(*, port: int = 0, enabled: bool = True) -> UISettings:
    return UISettings(
        enabled=enabled,
        host="127.0.0.1",
        port=port,
        max_episodes=10,
        redis_url="redis://127.0.0.1:6379/0",
        history_key_prefix="test:history",
    )


@contextmanager
def _running_server(monkeypatch, *, runtime=None, store=None, control_metadata=None, request_model_reset=None):
    if runtime is not None:
        monkeypatch.setattr(server_module, "get_runtime", lambda: runtime)
    if store is not None:
        monkeypatch.setattr(server_module, "get_store", lambda: store)
    if control_metadata is not None:
        monkeypatch.setattr(server_module, "control_metadata", control_metadata)
    if request_model_reset is not None:
        monkeypatch.setattr(server_module, "request_model_reset", request_model_reset)

    server = DashboardServer(_ui_settings())
    server.start()
    try:
        yield server
    finally:
        server.stop()


def _get(server: DashboardServer, path: str, headers: dict | None = None) -> http.client.HTTPResponse:
    host, port = server._httpd.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        response.body = response.read()
        return response
    finally:
        connection.close()


def _post(server: DashboardServer, path: str, *, headers: dict, body: bytes) -> http.client.HTTPResponse:
    host, port = server._httpd.server_address[:2]
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        response.body = response.read()
        return response
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_source_language_covers_every_known_extension():
    assert _source_language("runner.py") == "python"
    assert _source_language("config.yaml") == "yaml"
    assert _source_language("config.yml") == "yaml"
    assert _source_language("pyproject.toml") == "toml"
    assert _source_language("Containerfile") == "dockerfile"
    assert _source_language(".containerignore") == "text"


def test_generated_source_raises_for_a_file_outside_the_allowlist(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        generated_source("not-allowed.py", tmp_path)


def test_generated_sources_reports_language_and_size_for_every_present_file(tmp_path: Path):
    (tmp_path / "runner.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "Containerfile").write_text("FROM scratch\n", encoding="utf-8")

    files = {entry["path"]: entry for entry in generated_sources(tmp_path)}

    assert files["runner.py"]["language"] == "python"
    assert files["Containerfile"]["language"] == "dockerfile"


def test_redact_config_secrets_only_redacts_the_upload_section():
    content = 'upload:\n  api_key: "secret"\ntraining:\n  api_key: "not-a-secret"\n'

    redacted = _redact_config_secrets(content)

    assert 'upload:\n  api_key: "[REDACTED]"\n' in redacted
    assert 'training:\n  api_key: "not-a-secret"\n' in redacted


def test_rollout_videos_skips_malformed_and_incomplete_entries(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(record_dir=tmp_path)
    monkeypatch.setattr(server_module, "get_runtime", lambda: runtime)

    good = tmp_path / "good.rollout.json"
    good.write_text(json.dumps({"recorded_at": "2024-01-02T00:00:00Z", "score": 1.0}), encoding="utf-8")
    (tmp_path / "good.mp4").write_bytes(b"video")

    older = tmp_path / "older.rollout.json"
    older.write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z", "score": 1.0}), encoding="utf-8")
    (tmp_path / "older.mp4").write_bytes(b"video")

    (tmp_path / "missing-video.rollout.json").write_text(json.dumps({"score": 1.0}), encoding="utf-8")

    (tmp_path / "not-a-dict.rollout.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    (tmp_path / "not-a-dict.mp4").write_bytes(b"video")

    (tmp_path / "corrupt.rollout.json").write_text("not-json", encoding="utf-8")

    videos = rollout_videos()

    names = [entry["path"] for entry in videos]
    assert names == ["good.mp4", "older.mp4"]


def test_rollout_video_path_rejects_wrong_suffix_traversal_and_missing_sidecar(tmp_path: Path, monkeypatch):
    runtime = SimpleNamespace(record_dir=tmp_path)
    monkeypatch.setattr(server_module, "get_runtime", lambda: runtime)

    (tmp_path / "video.txt").write_bytes(b"not a video")
    with pytest.raises(FileNotFoundError):
        rollout_video_path("video.txt")

    outside = tmp_path.parent / "outside.mp4"
    outside.write_bytes(b"video")
    try:
        with pytest.raises(FileNotFoundError):
            rollout_video_path("../outside.mp4")
    finally:
        outside.unlink(missing_ok=True)

    (tmp_path / "no-sidecar.mp4").write_bytes(b"video")
    with pytest.raises(FileNotFoundError):
        rollout_video_path("no-sidecar.mp4")

    (tmp_path / "with-sidecar.mp4").write_bytes(b"video")
    (tmp_path / "with-sidecar.rollout.json").write_text("{}", encoding="utf-8")
    assert rollout_video_path("with-sidecar.mp4") == (tmp_path / "with-sidecar.mp4").resolve()


def test_datenwissenschaften_version_falls_back_when_package_metadata_is_missing(monkeypatch):
    def raise_not_found(_name):
        raise server_module.PackageNotFoundError

    monkeypatch.setattr(server_module, "version", raise_not_found)

    assert server_module._datenwissenschaften_version() == "DEVELOPMENT"


# ---------------------------------------------------------------------------
# DashboardServer / _DashboardHandler over real HTTP
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok(monkeypatch):
    with _running_server(monkeypatch) as server:
        response = _get(server, "/api/health")

    assert response.status == 200
    assert json.loads(response.body) == {"status": "ok"}


def test_snapshot_endpoint_merges_control_and_server_metadata(monkeypatch):
    store = SimpleNamespace(snapshot=lambda: {"summary": {}, "metadata": {}})
    with _running_server(
        monkeypatch,
        store=store,
        control_metadata=lambda: {"game": "Game", "restart_supported": True, "reset_pending": False},
    ) as server:
        response = _get(server, "/api/snapshot")
        payload = json.loads(response.body)

    assert response.status == 200
    assert payload["control"]["game"] == "Game"
    assert "csrf_token" in payload["control"]
    assert payload["server"]["bind_address"] == "127.0.0.1:0"


def test_sources_endpoint_lists_generated_files(monkeypatch, tmp_path: Path):
    (tmp_path / "runner.py").write_text("print(1)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with _running_server(monkeypatch) as server:
        response = _get(server, "/api/sources")
        payload = json.loads(response.body)

    assert response.status == 200
    assert payload["files"] == [{"path": "runner.py", "language": "python", "size": 9}]


def test_source_endpoint_returns_content_and_404s_for_unknown_paths(monkeypatch, tmp_path: Path):
    (tmp_path / "runner.py").write_text("print(1)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    with _running_server(monkeypatch) as server:
        ok = _get(server, "/api/source?path=runner.py")
        missing = _get(server, "/api/source?path=not-allowed.py")

    assert ok.status == 200
    assert json.loads(ok.body)["content"] == "print(1)\n"
    assert missing.status == 404


def test_rollout_videos_endpoint_lists_available_recordings(monkeypatch, tmp_path: Path):
    runtime = SimpleNamespace(record_dir=tmp_path)
    metadata_path = tmp_path / "episode.rollout.json"
    metadata_path.write_text(json.dumps({"recorded_at": "2024-01-01T00:00:00Z", "score": 1.0}), encoding="utf-8")
    (tmp_path / "episode.mp4").write_bytes(b"0123456789")

    with _running_server(monkeypatch, runtime=runtime) as server:
        response = _get(server, "/api/rollout-videos")
        payload = json.loads(response.body)

    assert response.status == 200
    assert payload["videos"][0]["path"] == "episode.mp4"


def test_rollout_video_endpoint_serves_full_and_partial_content(monkeypatch, tmp_path: Path):
    runtime = SimpleNamespace(record_dir=tmp_path)
    (tmp_path / "episode.rollout.json").write_text("{}", encoding="utf-8")
    (tmp_path / "episode.mp4").write_bytes(b"0123456789")

    with _running_server(monkeypatch, runtime=runtime) as server:
        full = _get(server, "/api/rollout-video?path=episode.mp4")
        partial = _get(server, "/api/rollout-video?path=episode.mp4", headers={"Range": "bytes=2-4"})
        open_ended = _get(server, "/api/rollout-video?path=episode.mp4", headers={"Range": "bytes=8-"})
        bad_range = _get(server, "/api/rollout-video?path=episode.mp4", headers={"Range": "bytes=9999-10000"})
        wrong_unit = _get(server, "/api/rollout-video?path=episode.mp4", headers={"Range": "items=0-3"})
        missing = _get(server, "/api/rollout-video?path=missing.mp4")

    assert full.status == 200
    assert full.body == b"0123456789"
    assert partial.status == 206
    assert partial.body == b"234"
    assert partial.getheader("Content-Range") == "bytes 2-4/10"
    assert open_ended.status == 206
    assert open_ended.body == b"89"
    assert bad_range.status == 416
    assert wrong_unit.status == 416
    assert missing.status == 404


def test_model_reset_endpoint_validates_csrf_content_type_and_payload(monkeypatch):
    with _running_server(monkeypatch, request_model_reset=lambda game: None) as server:
        csrf_token = server._httpd.csrf_token
        headers = {"Content-Type": "application/json", "X-CSRF-Token": csrf_token}

        wrong_csrf = _post(
            server,
            "/api/model/reset",
            headers={"Content-Type": "application/json", "X-CSRF-Token": "wrong"},
            body=json.dumps({"game": "Game"}).encode("utf-8"),
        )
        wrong_content_type = _post(
            server,
            "/api/model/reset",
            headers={"Content-Type": "text/plain", "X-CSRF-Token": csrf_token},
            body=b"game=Game",
        )
        bad_json = _post(server, "/api/model/reset", headers=headers, body=b"not-json")
        missing_game = _post(server, "/api/model/reset", headers=headers, body=json.dumps({}).encode("utf-8"))
        too_large = _post(
            server,
            "/api/model/reset",
            headers=headers,
            body=json.dumps({"game": "x" * 5000}).encode("utf-8"),
        )
        not_found = _post(server, "/api/other", headers=headers, body=b"{}")
        success = _post(server, "/api/model/reset", headers=headers, body=json.dumps({"game": "Game"}).encode("utf-8"))

    assert wrong_csrf.status == 403
    assert wrong_content_type.status == 415
    assert bad_json.status == 400
    assert missing_game.status == 400
    assert too_large.status == 400
    assert not_found.status == 404
    assert success.status == 202
    assert json.loads(success.body) == {"status": "reset_pending", "game": "Game"}


def test_model_reset_endpoint_reports_conflict_as_409(monkeypatch):
    def raise_runtime_error(game):
        raise RuntimeError("no active model")

    with _running_server(monkeypatch, request_model_reset=raise_runtime_error) as server:
        headers = {"Content-Type": "application/json", "X-CSRF-Token": server._httpd.csrf_token}
        response = _post(
            server,
            "/api/model/reset",
            headers=headers,
            body=json.dumps({"game": "Game"}).encode("utf-8"),
        )

    assert response.status == 409
    assert "no active model" in json.loads(response.body)["error"]


def test_static_asset_serving_falls_back_to_index_and_rejects_traversal(monkeypatch, tmp_path: Path):
    static_root = tmp_path / "static"
    static_root.mkdir()
    (static_root / "index.html").write_text("<html>dashboard</html>", encoding="utf-8")
    (static_root / "app.js").write_text("console.log(1);", encoding="utf-8")
    monkeypatch.setattr(server_module, "files", lambda _package: tmp_path)

    with _running_server(monkeypatch) as server:
        index = _get(server, "/")
        asset = _get(server, "/app.js")
        unknown = _get(server, "/unknown-route")
        traversal = _get(server, "/../secret")

    assert index.status == 200
    assert b"dashboard" in index.body
    assert asset.status == 200
    assert asset.getheader("Content-Type").startswith("text/javascript") or asset.getheader("Content-Type").startswith(
        "application/javascript"
    )
    assert unknown.status == 200
    assert b"dashboard" in unknown.body
    assert traversal.status == 404


def test_static_asset_serving_404s_when_assets_are_not_installed(monkeypatch):
    with _running_server(monkeypatch) as server:
        response = _get(server, "/")

    assert response.status == 404


def test_handler_ignores_client_disconnects_but_not_other_errors(monkeypatch):
    from datenwissenschaften.ui.server import _DashboardHandler

    handler = object.__new__(_DashboardHandler)
    monkeypatch.setattr(
        server_module.BaseHTTPRequestHandler,
        "handle",
        lambda _self: (_ for _ in ()).throw(ConnectionError()),
    )
    handler.handle()

    monkeypatch.setattr(
        server_module.BaseHTTPRequestHandler,
        "handle",
        lambda _self: (_ for _ in ()).throw(ValueError()),
    )
    with pytest.raises(ValueError):
        handler.handle()


# ---------------------------------------------------------------------------
# start_ui / global singleton
# ---------------------------------------------------------------------------


def test_start_ui_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr(server_module, "_server", None)

    assert server_module.start_ui(_ui_settings(enabled=False)) is None


def test_start_ui_starts_and_reuses_the_singleton_server(monkeypatch):
    monkeypatch.setattr(server_module, "_server", None)
    store = SimpleNamespace(resize=lambda _max_episodes: None)
    monkeypatch.setattr(server_module, "get_store", lambda: store)

    server = server_module.start_ui(_ui_settings())
    try:
        again = server_module.start_ui(_ui_settings())
        assert again is server
    finally:
        server.stop()
        monkeypatch.setattr(server_module, "_server", None)


def test_start_ui_returns_none_and_logs_when_the_port_is_taken(monkeypatch):
    monkeypatch.setattr(server_module, "_server", None)
    store = SimpleNamespace(resize=lambda _max_episodes: None)
    monkeypatch.setattr(server_module, "get_store", lambda: store)

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen(1)
        taken_port = blocker.getsockname()[1]

        result = server_module.start_ui(_ui_settings(port=taken_port))

    assert result is None
    assert server_module._server is None


def test_dashboard_server_start_and_stop_lifecycle():
    server = DashboardServer(_ui_settings())
    server.start()
    server.stop()
