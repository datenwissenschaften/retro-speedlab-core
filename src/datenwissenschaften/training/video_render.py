import subprocess
import sys
from pathlib import Path

import stable_retro as retro

PLAYBACK_MODULE = "datenwissenschaften.training.video_playback"


def render_video(roms_path: Path, recording: Path) -> Path:
    video = recording.with_suffix(".mp4")
    if video.is_file():
        return video
    playback = [sys.executable, "-m", PLAYBACK_MODULE, "--roms-dir", str(roms_path), "--no-audio", str(recording)]
    subprocess.run(playback, check=True, capture_output=True, text=True)
    return video


def count_frames(recording: Path) -> int:
    movie = retro.Movie(str(recording))
    frames = 0
    while movie.step():
        frames += 1
    return frames
