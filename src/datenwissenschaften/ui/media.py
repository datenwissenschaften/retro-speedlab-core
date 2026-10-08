import hashlib
import json
from typing import Any

MEDIA_KEY_BYTES = 16
VIDEO_SUFFIX = ".mp4"
STATUSES_SUFFIX = ".json"


def media_key(body: bytes) -> str:
    return hashlib.blake2b(body, digest_size=MEDIA_KEY_BYTES).hexdigest()


def episode_key(episode: dict[str, Any]) -> str:
    return media_key(episode["video"] + _statuses(episode))


def episode_media_names(episode: dict[str, Any]) -> set[str]:
    return {episode["key"] + VIDEO_SUFFIX, episode["key"] + STATUSES_SUFFIX}


def episode_media(episode: dict[str, Any], name: str) -> bytes:
    if name == episode["key"] + VIDEO_SUFFIX:
        return episode["video"]
    if name == episode["key"] + STATUSES_SUFFIX:
        return _statuses(episode)
    raise KeyError(name)


def _statuses(episode: dict[str, Any]) -> bytes:
    return json.dumps(episode["statuses"], separators=(",", ":")).encode("utf-8")
