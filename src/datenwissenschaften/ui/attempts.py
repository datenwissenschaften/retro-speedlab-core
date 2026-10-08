import json
from pathlib import Path
from typing import Any

from datenwissenschaften.ui.media import VIDEO_SUFFIX, media_key

METADATA_SUFFIX = ".rollout.json"


def recorded_attempts(record_root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    attempts = {}
    for metadata_path in record_root.glob(f"**/*{METADATA_SUFFIX}"):
        video = metadata_path.with_name(metadata_path.name.removesuffix(METADATA_SUFFIX) + VIDEO_SUFFIX)
        if not video.is_file():
            continue
        status = video.stat()
        key = media_key(f"{video}:{status.st_mtime_ns}:{status.st_size}".encode())
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        attempts[key + VIDEO_SUFFIX] = ({**metadata, "key": key}, video)
    return attempts
