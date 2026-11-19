import fcntl
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir

SLOTS = Path(gettempdir()) / "speedlab-encoders"
POLL_SECONDS = 0.5


@contextmanager
def encoder_slot(slots: int) -> Iterator[None]:
    SLOTS.mkdir(parents=True, exist_ok=True)
    while True:
        for slot in range(slots):
            with (SLOTS / f"{slot}.lock").open("w") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                yield
                return
        time.sleep(POLL_SECONDS)
