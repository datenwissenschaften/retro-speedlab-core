import subprocess
import sys
from pathlib import Path

from speedlab_engine.video import ENCODING

from datenwissenschaften.settings import RetroSpeedlabPaths

RENDERER = "speedlab_engine.cli"
MP4 = ".mp4"


def render_video(paths: RetroSpeedlabPaths, movies: list[Path]) -> list[Path]:
    for movie in movies:
        command = [
            sys.executable,
            "-m",
            RENDERER,
            "video",
            str(paths.collection),
            str(movie),
            str(movie.with_suffix(MP4)),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
    return [movie.with_suffix(MP4) for movie in movies]


def stack_videos(left: Path, right: Path, stacked: Path) -> Path:
    command = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(left), "-i", str(right)]
    command += ["-filter_complex", "hstack=inputs=2", *ENCODING, str(stacked)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return stacked
