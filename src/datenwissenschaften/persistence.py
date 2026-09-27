import json
import threading
from pathlib import Path
from typing import Any


class JsonDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}

    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._data

    def get(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._write()

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)
            self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._data, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self.path)
