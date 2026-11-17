import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

from datenwissenschaften.ui.relay import relay_client

PART_BYTES = 8 * 1024 * 1024
VIDEO_BYTES = 200 * 1024 * 1024
MP4 = {"Content-Type": "video/mp4"}
PART_ATTEMPTS = 6
FIRST_WAIT_SECONDS = 2.0
UPLOADING = threading.Lock()


@dataclass(slots=True, frozen=True)
class VideoUpload:
    url: str
    api_key: str
    name: str

    def put(self, video: Path, version: str, attachments: dict[str, dict[str, Any]]) -> None:
        body = video.read_bytes()
        if len(body) > VIDEO_BYTES:
            raise ValueError(f"The video {video} has {len(body)} bytes, more than {VIDEO_BYTES}")
        starts = range(0, len(body), PART_BYTES)
        with UPLOADING, relay_client(self.url, self.api_key) as client:
            if not self.held(client, version):
                for index, start in enumerate(starts):
                    self.part(client, index, body[start : start + PART_BYTES], index == len(starts) - 1)
            for name, document in attachments.items():
                client.put(f"/{name}", json={"version": version, **document}).raise_for_status()

    def held(self, client: httpx.Client, version: str) -> bool:
        response = client.head(f"/{self.name}")
        if response.status_code == HTTPStatus.NOT_FOUND:
            return False
        response.raise_for_status()
        return response.headers["etag"] == f'"{version}"'

    def part(self, client: httpx.Client, index: int, part: bytes, last: bool) -> None:
        for attempt in range(PART_ATTEMPTS - 1):
            try:
                self.send(client, index, part, last)
                return
            except httpx.TransportError as error:
                wait = FIRST_WAIT_SECONDS * 2**attempt
                logger.warning(f"Part {index} of the {self.name} video is sent again in {wait:.0f} s: {error!r}")
                time.sleep(wait)
        self.send(client, index, part, last)

    def send(self, client: httpx.Client, index: int, part: bytes, last: bool) -> None:
        client.put(f"/{self.name}/parts/{index}", params={"last": last}, content=part, headers=MP4).raise_for_status()
