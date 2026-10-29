import threading
from typing import Any

_lock = threading.Lock()
_metadata: dict[str, Any] = {}


def publish_metadata(section: str, values: dict[str, Any]) -> None:
    with _lock:
        _metadata[section] = values


def metadata() -> dict[str, Any]:
    with _lock:
        return dict(_metadata)
