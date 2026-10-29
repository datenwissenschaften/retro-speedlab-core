from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datenwissenschaften.ui.relay import backend_client

PART_BYTES = 8 * 1024 * 1024
VIDEO_BYTES = 200 * 1024 * 1024
MP4 = {"Content-Type": "video/mp4"}


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
        with backend_client(self.url, self.api_key) as client:
            for index, start in enumerate(starts):
                last = index == len(starts) - 1
                part = body[start : start + PART_BYTES]
                path = f"/{self.name}/parts/{index}"
                client.put(path, params={"last": last}, content=part, headers=MP4).raise_for_status()
            for name, document in attachments.items():
                client.put(f"/{name}", json={"version": version, **document}).raise_for_status()
