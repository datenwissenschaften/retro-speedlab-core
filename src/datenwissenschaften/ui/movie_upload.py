from dataclasses import dataclass
from pathlib import Path

from datenwissenschaften.ui.relay import backend_client

PART_BYTES = 8 * 1024 * 1024
MOVIE_BYTES = 200 * 1024 * 1024


@dataclass(slots=True, frozen=True)
class MovieUpload:
    url: str
    api_key: str

    def put(self, movie: Path) -> None:
        body = movie.read_bytes()
        if len(body) > MOVIE_BYTES:
            raise ValueError(f"The movie {movie} has {len(body)} bytes, more than {MOVIE_BYTES}")
        starts = range(0, len(body), PART_BYTES)
        with backend_client(self.url, self.api_key) as client:
            for index, start in enumerate(starts):
                part = body[start : start + PART_BYTES]
                last = index == len(starts) - 1
                headers = {"Content-Type": "video/mp4"}
                client.put(
                    f"/movie.mp4/parts/{index}", params={"last": last}, content=part, headers=headers
                ).raise_for_status()
