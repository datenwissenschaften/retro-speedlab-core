import subprocess
import sys
from pathlib import Path

from speedlab_engine.video import ENCODINGS

from datenwissenschaften.settings import RetroSpeedlabPaths, VideoSettings
from datenwissenschaften.speedrun.encoders import encoder_slot

RENDERER = "speedlab_engine.cli"
MP4 = ".mp4"


def render_video(paths: RetroSpeedlabPaths, video: VideoSettings, movies: list[Path]) -> list[Path]:
    for movie in movies:
        command = [
            sys.executable,
            "-m",
            RENDERER,
            "video",
            str(paths.collection),
            str(movie),
            str(movie.with_suffix(MP4)),
            video.encoder,
        ]
        with encoder_slot(video.max_encoders):
            subprocess.run(command, check=True, capture_output=True, text=True)
    return [movie.with_suffix(MP4) for movie in movies]


def stack_videos(left: Path, right: Path, stacked: Path, video: VideoSettings) -> Path:
    command = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(left), "-i", str(right)]
    command += ["-filter_complex", "hstack=inputs=2", *ENCODINGS[video.encoder], str(stacked)]
    with encoder_slot(video.max_encoders):
        subprocess.run(command, check=True, capture_output=True, text=True)
    return stacked
