import hashlib
from pathlib import Path

import msgspec
import numpy as np
from loguru import logger

from datenwissenschaften.reference.explored import archived_starts
from datenwissenschaften.reference.samples import current_level
from datenwissenschaften.reference.thumbnail import THUMB_SHAPE, thumbnail
from datenwissenschaften.speedrun.paths import atomic_write
from datenwissenschaften.speedrun.runner import SegmentRunner, Start


class Screenshot(msgspec.Struct, array_like=True, frozen=True):
    digest: str
    state: str
    level: int
    area: int
    x: int
    y: int
    dynamics: list[int]
    thumb: bytes

    def image(self) -> np.ndarray:
        return np.frombuffer(self.thumb, dtype=np.uint8).reshape(THUMB_SHAPE)


def digest_of(start: Start) -> str:
    return hashlib.sha256(b"|".join([start.state.encode(), start.emulator_state])).hexdigest()


def screenshot(runner: SegmentRunner, start: Start) -> Screenshot | None:
    runner.restore(start)
    runner.press(np.zeros(len(runner.emulator.buttons), dtype=np.int8))
    position = runner.ram.position()
    if position is None:
        return None
    thumb = thumbnail(runner.emulator.em.get_screen()).tobytes()
    level, dynamics = current_level(runner), list(runner.ram.dynamics())
    return Screenshot(digest_of(start), start.state, level, position.area, position.x, position.y, dynamics, thumb)


def load_screens(path: Path) -> dict[str, Screenshot]:
    if not path.is_file():
        return {}
    return {shot.digest: shot for shot in msgspec.msgpack.decode(path.read_bytes(), type=list[Screenshot])}


def updated_screens(runner: SegmentRunner, archives: Path, path: Path) -> list[Screenshot]:
    known = load_screens(path)
    fresh = [start for start in archived_starts(archives) if digest_of(start) not in known]
    for start in fresh:
        shot = screenshot(runner, start)
        if shot is not None:
            known[shot.digest] = shot
    atomic_write(path, msgspec.msgpack.encode(list(known.values())))
    logger.info(f"{len(fresh)} archived places were rendered; {len(known)} emulator screens face the video")
    return list(known.values())
