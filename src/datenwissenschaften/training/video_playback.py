import argparse
import subprocess
from pathlib import Path

import stable_retro as retro

from datenwissenschaften.environment.recording import recorded_buttons, restore_emulator_state
from datenwissenschaften.roms import import_roms

CONSTANT_RATE_FACTOR = "23"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roms-dir", type=Path, required=True)
    parser.add_argument("--integrations-dir", type=Path, required=True)
    parser.add_argument("movies", nargs="+", type=Path)
    args = parser.parse_args()
    import_roms(args.roms_dir, args.integrations_dir)
    for movie in args.movies:
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
    main()
