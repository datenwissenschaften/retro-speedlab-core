import io
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

Checkpoint = dict[str, Any]


class ModelStore:
    def __init__(self) -> None:
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-store")
        self.writing: Future[None] | None = None

    def read(self, path: Path) -> Future[Checkpoint | None]:
        return self.worker.submit(_read, path)

    def write(self, path: Path, serialize: Callable[[], io.BytesIO]) -> None:
        self._finish_writing()
        self.writing = self.worker.submit(_write, path, serialize())

    def close(self) -> None:
        self.worker.shutdown(wait=True)
        self._finish_writing()

    def _finish_writing(self) -> None:
        if self.writing is not None:
            self.writing.result()
            self.writing = None


def _read(path: Path) -> Checkpoint | None:
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu")


def _write(path: Path, checkpoint: io.BytesIO) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(checkpoint.getbuffer())
    temporary.replace(path)
