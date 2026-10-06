import subprocess
import tempfile
from pathlib import Path

CONSTANT_RATE_FACTOR = "26"


def encode_video(jpegs: list[bytes], frame_rate: float) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        video = Path(directory) / "replay.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-f",
                "image2pipe",
                "-framerate",
                f"{frame_rate:.6f}",
                "-c:v",
                "mjpeg",
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
                str(video),
            ],
            input=b"".join(jpegs),
            check=True,
            capture_output=True,
        )
        return video.read_bytes()
