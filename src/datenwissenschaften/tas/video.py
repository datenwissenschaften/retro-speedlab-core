import subprocess
import sys
from pathlib import Path

import stable_retro as retro
import typer

from datenwissenschaften.environment.recording import recorded_buttons, restore_emulator_state
from datenwissenschaften.roms import import_roms
from datenwissenschaften.settings import RetroSpeedlabPaths

CONSTANT_RATE_FACTOR = "26"
PIXEL_SCALE = 2
PLAYBACK_MODULE = "datenwissenschaften.tas.video"


def render_video(paths: RetroSpeedlabPaths, recording: Path) -> Path:
    playback = [
        sys.executable,
        "-m",
        PLAYBACK_MODULE,
        str(paths.roms_path),
        str(paths.integrations_dir),
        str(recording),
    ]
    subprocess.run(playback, check=True, capture_output=True, text=True)
    return recording.with_suffix(".mp4")


def main(roms_dir: Path, integrations_dir: Path, movies: list[Path]) -> None:
    import_roms(roms_dir, integrations_dir)
    for movie in movies:
        render(movie)


def render(recording: Path) -> None:
    movie = retro.Movie(str(recording))
    env = retro.make(
        movie.get_game(), retro.State.NONE, render_mode="rgb_array", use_restricted_actions=retro.Actions.ALL
    )
    try:
        frame, _ = env.reset()
        emulator = env.unwrapped
        restore_emulator_state(emulator, movie.get_state())
        height, width, _ = frame.shape
        encoder = subprocess.Popen(
            encoder_command(width, height, emulator.em.get_screen_rate(), recording), stdin=subprocess.PIPE
        )
        for buttons in recorded_buttons(movie, emulator.num_buttons):
            frame, *_ = env.step(buttons)
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to encode {recording.name}")
    finally:
        env.close()


def encoder_command(width: int, height: int, frame_rate: float, recording: Path) -> list[str]:
    return [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        f"{frame_rate:.6f}",
        "-i",
        "pipe:",
        "-vf",
        f"scale=iw*{PIXEL_SCALE}:ih*{PIXEL_SCALE}:flags=neighbor",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        CONSTANT_RATE_FACTOR,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(recording.with_suffix(".mp4")),
    ]


if __name__ == "__main__":
    typer.run(main)
