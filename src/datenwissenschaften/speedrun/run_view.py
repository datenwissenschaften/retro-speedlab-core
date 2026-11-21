import subprocess
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import msgspec
import numpy as np
import typer
from numpy.typing import NDArray
from speedlab_engine.movie import Movie
from speedlab_engine.soundtrack import mix
from speedlab_engine.video import ENCODINGS, Encoder, audio_command, cues_of

from datenwissenschaften.environment.factory import make_environment
from datenwissenschaften.settings import load_config
from datenwissenschaften.speedrun.encoders import encoder_slot
from datenwissenschaften.speedrun.journal import Split
from datenwissenschaften.speedrun.movie import Segment, memory_sample, pieces_pressed, recorded
from datenwissenschaften.speedrun.report import Course, course_of, splits
from datenwissenschaften.speedrun.run_panel import HEIGHT, WIDTH, Moment, annotated
from datenwissenschaften.speedrun.runner import SegmentRunner
from datenwissenschaften.workflow.wrapper import wrapper_class

app = typer.Typer()


def playing(found: list[Split], first: str) -> str:
    return found[-1].level if found else first


def moments(
    runner: SegmentRunner, course: Course, segments: list[Segment]
) -> Iterator[tuple[NDArray[np.uint8], Moment]]:
    found = splits(course, segments)
    runner.restore(runner.power_on())
    for frame, buttons in enumerate(np.asarray(pieces_pressed(segments), dtype=np.int8)):
        runner.emulator.step(buttons)
        seconds = (frame + 1) / course.frame_rate
        cleared = [split for split in found if split.seconds <= seconds]
        level = found[len(cleared)].level if len(cleared) < len(found) else playing(found, course.first)
        memory = memory_sample(runner.wrapper.read_ram())
        yield (
            np.asarray(runner.emulator.em.get_screen(), dtype=np.uint8),
            Moment(frame, seconds, level, cleared, memory),
        )


def encoder_command(
    video: Path, frame_rate: float, soundtrack: tuple[Path, Path] | None, encoder: Encoder
) -> list[str]:
    command = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    command += ["-s", f"{WIDTH}x{HEIGHT}", "-framerate", f"{frame_rate:.6f}", "-i", "pipe:"]
    if soundtrack is not None:
        command += audio_command(soundtrack)
    return [*command, *ENCODINGS[encoder], str(video)]


def soundtrack(
    runner: SegmentRunner, segments: list[Segment], collection: Path, folder: Path
) -> tuple[Path, Path] | None:
    movie = msgspec.msgpack.decode(recorded(runner, runner.power_on(), pieces_pressed(segments)), type=Movie)
    cues = cues_of(movie, collection)
    return None if cues is None else mix(cues, len(movie.frames), folder)


def render_run_view(wrapper: str, config: Path, pieces: Path, video: Path) -> Path:
    settings = load_config(config)
    runner = SegmentRunner(make_environment(wrapper_class(wrapper), settings))
    course = course_of(runner, settings.speedrun.game)
    segments = msgspec.msgpack.decode(pieces.read_bytes(), type=list[Segment])
    with TemporaryDirectory() as folder:
        sound = soundtrack(runner, segments, settings.paths.collection, Path(folder))
        command = encoder_command(video, course.frame_rate, sound, settings.video.encoder)
        with encoder_slot(settings.video.max_encoders), subprocess.Popen(command, stdin=subprocess.PIPE) as ffmpeg:
            if ffmpeg.stdin is None:
                raise RuntimeError("ffmpeg opened no input pipe")
            for screen, moment in moments(runner, course, segments):
                ffmpeg.stdin.write(annotated(screen, settings.speedrun.game.title, moment).tobytes())
            ffmpeg.stdin.close()
            if ffmpeg.wait() != 0:
                raise RuntimeError(f"ffmpeg failed to encode {video.name}")
    return video


@app.command()
def view(wrapper: str, config: Path, pieces: Path, video: Path) -> None:
    render_run_view(wrapper, config, pieces, video)


if __name__ == "__main__":
    app()
