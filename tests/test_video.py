from pathlib import Path

from fakes import write_config
from typer.testing import CliRunner

from datenwissenschaften.settings import load_config
from datenwissenschaften.tas import video


def test_a_movie_is_rendered_by_the_playback_module_next_to_it(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(video.subprocess, "run", lambda command, check, capture_output, text: commands.append(command))
    paths = load_config(write_config(tmp_path)).paths

    rendered = video.render_video(paths, [tmp_path / "tas.bk2"])

    assert rendered == [tmp_path / "tas.mp4"]
    assert commands[0][1:] == [
        "-m",
        video.PLAYBACK_MODULE,
        str(paths.roms_path),
        str(paths.integrations_dir),
        str(tmp_path / "tas.bk2"),
    ]


def test_the_playback_imports_the_roms_and_renders_every_movie(tmp_path: Path, monkeypatch):
    calls: list[object] = []
    monkeypatch.setattr(video, "import_roms", lambda roms, integrations: calls.append((roms, integrations)))
    monkeypatch.setattr(video, "render", calls.append)
    app = video.typer.Typer()
    app.command()(video.main)

    result = CliRunner().invoke(app, [str(tmp_path / "roms"), str(tmp_path / "integrations"), "a.bk2", "b.bk2"])

    assert result.exit_code == 0
    assert calls == [(tmp_path / "roms", tmp_path / "integrations"), Path("a.bk2"), Path("b.bk2")]


def test_the_encoder_writes_an_mp4_beside_the_movie():
    command = video.encoder_command(240, 224, 60.0, Path("run.bk2"))

    assert command[command.index("-s") + 1] == "240x224"
    assert command[-1] == "run.mp4"


def test_two_videos_are_stacked_side_by_side(tmp_path: Path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setattr(video.subprocess, "run", lambda command, check, capture_output, text: commands.append(command))

    stacked = video.stack_videos(tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "ab.mp4")

    assert stacked == tmp_path / "ab.mp4"
    assert commands[0][commands[0].index("-filter_complex") + 1] == "hstack=inputs=2"
