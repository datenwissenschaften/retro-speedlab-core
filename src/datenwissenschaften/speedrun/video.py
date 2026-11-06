import subprocess
import sys
from pathlib import Path

import stable_retro as retro
import typer

from datenwissenschaften.environment.recording import recorded_buttons, restore_emulator_state
from datenwissenschaften.knowledge import knowledge_path, load_knowledge
from datenwissenschaften.roms import import_roms
from datenwissenschaften.settings import RetroSpeedlabPaths
from datenwissenschaften.sprites.objects import objects_in
from datenwissenschaften.sprites.overlay import PIXEL_SCALE, Labels, annotate, labels_of

CONSTANT_RATE_FACTOR = "26"
PLAYBACK_MODULE = "datenwissenschaften.speedrun.video"
ENCODING = (
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
)


def render_video(paths: RetroSpeedlabPaths, recordings: list[Path]) -> list[Path]:
    playback = [
        sys.executable,
        "-m",
        PLAYBACK_MODULE,
        str(paths.roms_path),
        str(paths.integrations_dir),
        str(knowledge_path(paths.game_dir)),
        *map(str, recordings),
    ]
    subprocess.run(playback, check=True, capture_output=True, text=True)
    return [recording.with_suffix(".mp4") for recording in recordings]


def stack_videos(left: Path, right: Path, stacked: Path) -> Path:
    command = ["ffmpeg", "-loglevel", "error", "-y", "-i", str(left), "-i", str(right)]
    command += ["-filter_complex", "hstack=inputs=2", *ENCODING, str(stacked)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return stacked


def main(roms_dir: Path, integrations_dir: Path, knowledge: Path, movies: list[Path]) -> None:
    import_roms(roms_dir, integrations_dir)
    labels = labels_of(load_knowledge(knowledge))
    for movie in movies:
        render(movie, labels)


def render(recording: Path, labels: Labels) -> None:
    movie = retro.Movie(str(recording))
    env = retro.make(
        movie.get_game(), retro.State.NONE, render_mode="rgb_array", use_restricted_actions=retro.Actions.ALL
    )
    try:
        frame, _ = env.reset()
        restore_emulator_state(env, movie.get_state())
        height, width, _ = frame.shape
        size = (width * PIXEL_SCALE, height * PIXEL_SCALE)
        encoder = subprocess.Popen(encoder_command(*size, env.em.get_screen_rate(), recording), stdin=subprocess.PIPE)
        if encoder.stdin is None:
            raise RuntimeError("ffmpeg opened no input pipe")
        for buttons in recorded_buttons(movie, env.num_buttons):
            frame, *_ = env.step(buttons)
            encoder.stdin.write(annotate(frame, objects_in(env.em.get_state(), frame), labels).tobytes())
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
        *ENCODING,
        str(recording.with_suffix(".mp4")),
    ]


if __name__ == "__main__":
    typer.run(main)
